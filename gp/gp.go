package gp

import (
	"errors"
	"fmt"
	"math"

	"github.com/guoyyit/gaussian-process-regression-go/internal/linalg"
	"github.com/guoyyit/gaussian-process-regression-go/kernel"
)

// Prediction is a single-point GP predictive distribution.
type Prediction struct {
	Mean     float64
	Variance float64
	// ObservedVariance is Variance + sigma_n^2: the variance of a new
	// noisy observation y* = f* + eps at this point.
	ObservedVariance float64
}

var (
	errNoData      = errors.New("gp: training data is empty")
	errDimMismatch = errors.New("gp: input dimension mismatch")
	errBadNoise    = errors.New("gp: noise variance must be non-negative")
	errBadInput    = errors.New("gp: input contains NaN or Inf")
)

// Model is a trained (or training) Gaussian process regressor.
//
// A Model uses a constant zero mean function. Prediction variances are
// noise-inclusive: var* = k** - k*^T (K+sigma_n^2 I)^{-1} k* + sigma_n^2,
// so the variance at a noisy observation point tends toward sigma_n^2.
type Model struct {
	Kernel  kernel.Kernel
	X       [][]float64
	Y       []float64
	Options Options
	Noise   float64

	progress *OptimizeProgress

	// factorization cache, invalidated on parameter updates
	chol        [][]float64
	alpha       []float64
	appliedJit  float64
	cachedValid bool
}

// NewModel validates inputs and creates an untrained model.
func NewModel(k kernel.Kernel, x [][]float64, y []float64, opts Options) (*Model, error) {
	if len(x) == 0 || len(y) == 0 {
		return nil, errNoData
	}
	if len(x) != len(y) {
		return nil, fmt.Errorf("gp: %d inputs but %d targets", len(x), len(y))
	}
	if k == nil {
		return nil, errors.New("gp: kernel is nil")
	}
	dim := k.InputDim()
	for i, xi := range x {
		if len(xi) != dim {
			return nil, fmt.Errorf("%w: x[%d] has dim %d, want %d", errDimMismatch, i, len(xi), dim)
		}
		for _, v := range xi {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("%w: x[%d]", errBadInput, i)
			}
		}
		if math.IsNaN(y[i]) || math.IsInf(y[i], 0) {
			return nil, fmt.Errorf("%w: y[%d]", errBadInput, i)
		}
	}
	opts = opts.withDefaults()
	if opts.NoiseVariance < 0 || math.IsNaN(opts.NoiseVariance) || math.IsInf(opts.NoiseVariance, 0) {
		return nil, errBadNoise
	}
	return &Model{
		Kernel:  k,
		X:       x,
		Y:       y,
		Options: opts,
		Noise:   opts.NoiseVariance,
	}, nil
}

func (m *Model) invalidateCache() {
	m.cachedValid = false
	m.chol = nil
	m.alpha = nil
}

// covariance builds K_xx + noise*I (without jitter).
func (m *Model) covariance(noise float64) [][]float64 {
	n := len(m.X)
	kmat := make([][]float64, n)
	for i := 0; i < n; i++ {
		kmat[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := m.Kernel.Cov(m.X[i], m.X[j])
			kmat[i][j] = v
			kmat[j][i] = v
		}
		kmat[i][i] += noise
	}
	return kmat
}

// factorize builds K+noise*I, applies the configured stability policy and
// computes L and alpha = (K+noise I)^{-1} y.
func (m *Model) factorize() error {
	if m.cachedValid {
		return nil
	}
	kmat := m.covariance(m.Noise)
	l, err := linalg.Cholesky(kmat)
	jit := 0.0
	if err != nil {
		switch m.Options.Stability {
		case StabilityStrict:
			return fmt.Errorf("gp: kernel matrix not positive definite and stability=strict: %w", err)
		case StabilityJitter:
			jit = m.Options.InitialJitter
			for {
				if jit > m.Options.MaxJitter {
					return fmt.Errorf("gp: kernel matrix ill-conditioned beyond max jitter %g: %w", m.Options.MaxJitter, err)
				}
				jmat := linalg.CloneMatrix(kmat)
				for i := range jmat {
					jmat[i][i] += jit
				}
				l, err = linalg.Cholesky(jmat)
				if err == nil {
					break
				}
				jit *= m.Options.JitterFactor
			}
		default:
			return fmt.Errorf("gp: unknown stability strategy %q", m.Options.Stability)
		}
	}
	yb := make([][]float64, len(m.Y))
	for i, v := range m.Y {
		yb[i] = []float64{v}
	}
	solved := linalg.SolveCholesky(l, yb)
	alpha := make([]float64, len(m.Y))
	for i := range alpha {
		alpha[i] = solved[i][0]
		if math.IsNaN(alpha[i]) || math.IsInf(alpha[i], 0) {
			return errors.New("gp: solve produced NaN/Inf; kernel matrix is ill-conditioned")
		}
	}
	m.chol = l
	m.alpha = alpha
	m.appliedJit = jit
	m.cachedValid = true
	return nil
}

// crossCovariance returns k* between a query point and all training points.
func (m *Model) crossCovariance(x []float64) ([]float64, error) {
	if len(x) != m.Kernel.InputDim() {
		return nil, fmt.Errorf("%w: query has dim %d, want %d", errDimMismatch, len(x), m.Kernel.InputDim())
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, errBadInput
		}
	}
	kstar := make([]float64, len(m.X))
	for i := range m.X {
		kstar[i] = m.Kernel.Cov(x, m.X[i])
	}
	return kstar, nil
}

// Predict returns the predictive mean and latent-function posterior
// variance f* | X,y. At a noise-free observation point the variance
// approaches zero; with observation noise sigma_n^2 it approaches
// sigma_n^2 (within a documented tolerance).
//
// ObservedVariance adds the current noise variance, i.e. it predicts the
// variance of a new noisy measurement at the point.
func (m *Model) Predict(x []float64) (Prediction, error) {
	if err := m.factorize(); err != nil {
		return Prediction{}, err
	}
	kstar, err := m.crossCovariance(x)
	if err != nil {
		return Prediction{}, err
	}
	mean := 0.0
	for i := range m.X {
		mean += kstar[i] * m.alpha[i]
	}
	if math.IsNaN(mean) || math.IsInf(mean, 0) {
		return Prediction{}, errors.New("gp: predictive mean is NaN/Inf")
	}
	kb := make([][]float64, len(kstar))
	for i, v := range kstar {
		kb[i] = []float64{v}
	}
	v := linalg.SolveCholesky(m.chol, kb)
	quad := 0.0
	for i := range kstar {
		quad += kstar[i] * v[i][0]
	}
	kss := m.Kernel.Cov(x, x)
	variance := kss - quad
	// Clamp tiny negative round-off only; larger negatives surface errors.
	if variance < 0 {
		if variance > -1e-8*(kss+1) {
			variance = 0
		} else {
			return Prediction{}, fmt.Errorf("gp: predictive variance significantly negative (%g); model unstable", variance)
		}
	}
	if math.IsNaN(variance) || math.IsInf(variance, 0) {
		return Prediction{}, errors.New("gp: predictive variance is NaN/Inf")
	}
	return Prediction{Mean: mean, Variance: variance, ObservedVariance: variance + m.Noise}, nil
}

// PredictBatch predicts multiple points.
func (m *Model) PredictBatch(xs [][]float64) ([]Prediction, error) {
	out := make([]Prediction, len(xs))
	for i, x := range xs {
		p, err := m.Predict(x)
		if err != nil {
			return nil, fmt.Errorf("gp: prediction %d failed: %w", i, err)
		}
		out[i] = p
	}
	return out, nil
}

// logMarginalLikelihoodAt evaluates
//
//	-1/2 y^T K_y^{-1} y - 1/2 log|K_y| - n/2 log(2 pi)
//
// at the current kernel/noise parameters, together with the gradient
// with respect to the natural kernel parameters and the noise variance.
func (m *Model) logMarginalLikelihoodAt() (lml float64, grad []float64, noiseGrad float64, jit float64, err error) {
	if ferr := m.factorize(); ferr != nil {
		return 0, nil, 0, 0, ferr
	}
	n := len(m.X)
	quad := 0.0
	for i := range m.Y {
		quad += m.Y[i] * m.alpha[i]
	}
	logDet := linalg.DeterminantLog(m.chol)
	lml = -0.5*quad - 0.5*logDet - 0.5*float64(n)*math.Log(2*math.Pi)

	// a a^T - K_y^{-1}
	inv := linalg.SolveCholesky(m.chol, identity(n))
	np := m.Kernel.NumParams()
	grad = make([]float64, np)
	buf := make([]float64, np)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			m.Kernel.GradCov(m.X[i], m.X[j], buf)
			w := m.alpha[i]*m.alpha[j] - inv[i][j]
			mult := w
			if i != j {
				mult = 2 * w // symmetry: accumulate both (i,j) and (j,i)
			}
			for p := 0; p < np; p++ {
				grad[p] += 0.5 * mult * buf[p]
			}
		}
	}
	// d LML / d sigma_n^2 = 1/2 tr((a a^T - K_y^{-1}))
	tr := 0.0
	for i := 0; i < n; i++ {
		tr += m.alpha[i]*m.alpha[i] - inv[i][i]
	}
	noiseGrad = 0.5 * tr
	return lml, grad, noiseGrad, m.appliedJit, nil
}

// LogMarginalLikelihood evaluates the log marginal likelihood and its
// gradient (natural parameterization) at the current parameters.
func (m *Model) LogMarginalLikelihood() (lml float64, grad []float64, err error) {
	lml, grad, _, _, err = m.logMarginalLikelihoodAt()
	return lml, grad, err
}

// Progress returns the optimization progress recorded so far.
func (m *Model) Progress() *OptimizeProgress { return m.progress }

// SetProgress attaches an optimization-progress record to the model.
func (m *Model) SetProgress(p OptimizeProgress) {
	cp := p
	m.progress = &cp
}

// AppliedJitter returns the diagonal jitter applied during factorization.
func (m *Model) AppliedJitter() float64 { return m.appliedJit }

// Prepare eagerly computes the Cholesky factorization. Call this before
// publishing a model to concurrent readers so that the read path never
// lazily mutates shared state (Predict is then read-only).
func (m *Model) Prepare() error {
	return m.factorize()
}

func identity(n int) [][]float64 {
	id := make([][]float64, n)
	for i := range id {
		id[i] = make([]float64, n)
		id[i][i] = 1
	}
	return id
}
