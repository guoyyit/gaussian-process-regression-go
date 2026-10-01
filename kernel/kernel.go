package kernel

import (
	"errors"
	"fmt"
	"math"

	"github.com/guoyyit/gaussian-process-regression-go/internal/linalg"
)

// Type identifies a built-in covariance kernel.
type Type string

const (
	RBF      Type = "rbf"
	Matern32 Type = "matern32"
	Matern52 Type = "matern52"
	Linear   Type = "linear"
)

const (
	// DefaultSignalVariance is the default prior output variance.
	DefaultSignalVariance = 1.0
	// DefaultLengthScale is the default per-dimension length scale.
	DefaultLengthScale = 1.0
	// DefaultBiasVariance is the default linear-kernel constant term.
	DefaultBiasVariance = 1.0
	// DefaultScaleVariance is the default linear-kernel slope variance.
	DefaultScaleVariance = 1.0
)

// Config describes how to construct a kernel.
//
// RBF/Matern use LengthScales and SignalVariance (defaults applied on
// zero values). Linear uses BiasVariance and ScaleVariance.
type Config struct {
	Type           Type
	InputDim       int
	LengthScales   []float64
	SignalVariance float64
	BiasVariance   float64
	ScaleVariance  float64
}

// Kernel is a covariance function over fixed-dimension input points.
//
// Params are stored in their natural (positive) parameterization; the GP
// optimizer works with log-transformed parameters.
type Kernel interface {
	Name() string
	InputDim() int
	NumParams() int
	Params() []float64
	SetParams(p []float64) error
	Cov(x1, x2 []float64) float64
	// GradCov fills grad with d Cov / d param_k for every kernel
	// parameter, evaluated at x1, x2.
	GradCov(x1, x2 []float64, grad []float64)
	Clone() Kernel
}

var (
	errBadDim    = errors.New("kernel: input dimension mismatch")
	errBadParams = errors.New("kernel: parameter count or value invalid")
)

// New constructs a kernel from a Config.
func New(cfg Config) (Kernel, error) {
	if cfg.InputDim <= 0 {
		return nil, fmt.Errorf("%w: InputDim must be positive", errBadParams)
	}
	switch cfg.Type {
	case RBF:
		return newStationary(cfg, 0)
	case Matern32:
		return newStationary(cfg, 1.5)
	case Matern52:
		return newStationary(cfg, 2.5)
	case Linear:
		return newLinear(cfg), nil
	default:
		return nil, fmt.Errorf("kernel: unknown type %q", cfg.Type)
	}
}

func resolveLengthScales(cfg Config) []float64 {
	ls := make([]float64, cfg.InputDim)
	if len(cfg.LengthScales) == 1 {
		for d := 0; d < cfg.InputDim; d++ {
			ls[d] = cfg.LengthScales[0]
		}
	} else if len(cfg.LengthScales) == cfg.InputDim {
		copy(ls, cfg.LengthScales)
	} else if len(cfg.LengthScales) > 0 {
		return nil
	} else {
		for d := range ls {
			ls[d] = DefaultLengthScale
		}
	}
	for d := range ls {
		if !(ls[d] > 0) || math.IsInf(ls[d], 0) {
			return nil
		}
	}
	return ls
}

type stationaryKernel struct {
	name   string
	nu     float64 // 0 for RBF, 1.5 or 2.5 for Matern
	dim    int
	ls     []float64
	sigma2 float64
}

func newStationary(cfg Config, nu float64) (Kernel, error) {
	ls := resolveLengthScales(cfg)
	if ls == nil {
		return nil, fmt.Errorf("%w: length scales must be positive and match InputDim (or be scalar)", errBadParams)
	}
	sigma2 := cfg.SignalVariance
	if sigma2 == 0 {
		sigma2 = DefaultSignalVariance
	}
	if !(sigma2 > 0) || math.IsInf(sigma2, 0) {
		return nil, fmt.Errorf("%w: signal variance must be positive", errBadParams)
	}
	name := string(RBF)
	switch nu {
	case 1.5:
		name = string(Matern32)
	case 2.5:
		name = string(Matern52)
	}
	return &stationaryKernel{name: name, nu: nu, dim: cfg.InputDim, ls: ls, sigma2: sigma2}, nil
}

func (k *stationaryKernel) Name() string  { return k.name }
func (k *stationaryKernel) InputDim() int { return k.dim }
func (k *stationaryKernel) NumParams() int {
	return k.dim + 1 // log-length scales ... , signal variance
}

func (k *stationaryKernel) Params() []float64 {
	p := make([]float64, k.NumParams())
	copy(p, k.ls)
	p[k.dim] = k.sigma2
	return p
}

func (k *stationaryKernel) SetParams(p []float64) error {
	if len(p) != k.NumParams() {
		return fmt.Errorf("%w: got %d params, want %d", errBadParams, len(p), k.NumParams())
	}
	for i := 0; i < k.dim; i++ {
		if !(p[i] > 0) {
			return fmt.Errorf("%w: length scale must be positive", errBadParams)
		}
	}
	if !(p[k.dim] > 0) {
		return fmt.Errorf("%w: signal variance must be positive", errBadParams)
	}
	copy(k.ls, p[:k.dim])
	k.sigma2 = p[k.dim]
	return nil
}

func (k *stationaryKernel) check(x1, x2 []float64) error {
	if len(x1) != k.dim || len(x2) != k.dim {
		return errBadDim
	}
	return nil
}

// scaledDist2 returns sum_d ((x1_d-x2_d)/ls_d)^2 and the per-dimension
// squared scaled differences in dd.
func (k *stationaryKernel) scaledDist2(x1, x2 []float64, dd []float64) float64 {
	r2 := 0.0
	for d := 0; d < k.dim; d++ {
		z := (x1[d] - x2[d]) / k.ls[d]
		dd[d] = z * z
		r2 += dd[d]
	}
	return r2
}

func (k *stationaryKernel) Cov(x1, x2 []float64) float64 {
	if err := k.check(x1, x2); err != nil {
		return math.NaN()
	}
	dd := make([]float64, k.dim)
	r2 := k.scaledDist2(x1, x2, dd)
	return k.sigma2 * k.correlation(r2, math.Sqrt(r2))
}

// correlation returns the unit-variance correlation for distance r.
func (k *stationaryKernel) correlation(r2, r float64) float64 {
	switch k.nu {
	case 0:
		return math.Exp(-0.5 * r2)
	case 1.5:
		t := math.Sqrt(3.0) * r
		return (1 + t) * math.Exp(-t)
	case 2.5:
		t := math.Sqrt(5.0) * r
		return (1 + t + t*t/3.0) * math.Exp(-t)
	default:
		return math.NaN()
	}
}

// dcorrDR2 returns d correlation / d r2 where r2 is scaled distance
// squared. Derivations are closed form for RBF, Matern3/2 and Matern5/2.
func (k *stationaryKernel) dcorrDR2(r float64) float64 {
	switch k.nu {
	case 0:
		return -0.5 * math.Exp(-0.5*r*r)
	case 1.5:
		s3 := math.Sqrt(3.0)
		e := math.Exp(-s3 * r)
		// d/dr2 [(1+s3 r) e^{-s3 r}] = -3/2 e^{-s3 r}
		return -1.5 * e
	case 2.5:
		s5 := math.Sqrt(5.0)
		e := math.Exp(-s5 * r)
		// d/dr2 [(1+s5 r + (s5 r)^2/3) e^{-s5 r}]
		// = -(5/6)(1+s5 r) e^{-s5 r}
		return -(5.0 / 6.0) * (1 + s5*r) * e
	default:
		return math.NaN()
	}
}

func (k *stationaryKernel) GradCov(x1, x2 []float64, grad []float64) {
	if err := k.check(x1, x2); err != nil || len(grad) != k.NumParams() {
		for i := range grad {
			grad[i] = math.NaN()
		}
		return
	}
	dd := make([]float64, k.dim)
	r2 := k.scaledDist2(x1, x2, dd)
	r := math.Sqrt(r2)
	corr := k.correlation(r2, r)
	// d r2 / d ls_d = -2 dd_d / ls_d ; combine with d corr / d r2.
	dc := k.dcorrDR2(r)
	for d := 0; d < k.dim; d++ {
		grad[d] = k.sigma2 * dc * (-2 * dd[d] / k.ls[d])
	}
	grad[k.dim] = corr // d Cov / d sigma2
}

func (k *stationaryKernel) Clone() Kernel {
	ls := append([]float64(nil), k.ls...)
	return &stationaryKernel{name: k.name, nu: k.nu, dim: k.dim, ls: ls, sigma2: k.sigma2}
}

type linearKernel struct {
	dim      int
	biasVar  float64
	scaleVar float64
}

func newLinear(cfg Config) *linearKernel {
	bias := cfg.BiasVariance
	if bias == 0 {
		bias = DefaultBiasVariance
	}
	scale := cfg.ScaleVariance
	if scale == 0 {
		scale = DefaultScaleVariance
	}
	return &linearKernel{dim: cfg.InputDim, biasVar: bias, scaleVar: scale}
}

func (k *linearKernel) Name() string   { return string(Linear) }
func (k *linearKernel) InputDim() int  { return k.dim }
func (k *linearKernel) NumParams() int { return 2 }

func (k *linearKernel) Params() []float64 {
	return []float64{k.biasVar, k.scaleVar}
}

func (k *linearKernel) SetParams(p []float64) error {
	if len(p) != 2 || !(p[0] > 0) || !(p[1] > 0) {
		return fmt.Errorf("%w: linear kernel needs two positive params [biasVar, scaleVar]", errBadParams)
	}
	k.biasVar, k.scaleVar = p[0], p[1]
	return nil
}

func (k *linearKernel) Cov(x1, x2 []float64) float64 {
	if len(x1) != k.dim || len(x2) != k.dim {
		return math.NaN()
	}
	return k.biasVar + k.scaleVar*linalg.Dot(x1, x2)
}

func (k *linearKernel) GradCov(x1, x2 []float64, grad []float64) {
	if len(x1) != k.dim || len(x2) != k.dim || len(grad) != 2 {
		for i := range grad {
			grad[i] = math.NaN()
		}
		return
	}
	grad[0] = 1
	grad[1] = linalg.Dot(x1, x2)
}

func (k *linearKernel) Clone() Kernel {
	return &linearKernel{dim: k.dim, biasVar: k.biasVar, scaleVar: k.scaleVar}
}
