package gp

import (
	"fmt"
	"math"
)

// Bounds for log-transformed hyperparameters, keeping natural values in
// [1e-10, 1e6]. These prevent pathological optimizer steps while leaving
// a wide usable range.
const (
	logParamLower = -23.02585092994046 // log(1e-10)
	logParamUpper = 13.815510557964274 // log(1e6)
	lbfgsMem      = 10
)

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// applyLogParams applies optimizer parameters (log space) to the model
// and invalidates the factorization cache.
func (m *Model) applyLogParams(z []float64, optimizeNoise bool) error {
	nk := m.Kernel.NumParams()
	kp := make([]float64, nk)
	for i := 0; i < nk; i++ {
		kp[i] = math.Exp(clamp(z[i], logParamLower, logParamUpper))
	}
	if err := m.Kernel.SetParams(kp); err != nil {
		return err
	}
	if optimizeNoise {
		m.Noise = math.Exp(clamp(z[nk], logParamLower, logParamUpper))
	}
	m.invalidateCache()
	return nil
}

// negLML evaluates -LML and its gradient w.r.t. log parameters.
// Chain rule: df/d log p = p * df/d p.
func (m *Model) negLML(z []float64, optimizeNoise bool) (f float64, gz []float64, jit float64, err error) {
	if err := m.applyLogParams(z, optimizeNoise); err != nil {
		return 0, nil, 0, err
	}
	lml, gk, noiseGrad, jit, err := m.logMarginalLikelihoodAt()
	if err != nil {
		return 0, nil, 0, err
	}
	nk := m.Kernel.NumParams()
	gz = make([]float64, nk+b2i(optimizeNoise))
	kparams := m.Kernel.Params()
	for i := 0; i < nk; i++ {
		gz[i] = -kparams[i] * gk[i]
	}
	if optimizeNoise {
		gz[nk] = -m.Noise * noiseGrad
	}
	return -lml, gz, jit, nil
}

// Train fits hyperparameters by maximizing the log marginal likelihood.
//
// It uses L-BFGS with an Armijo backtracking line search. The run stops
// on relative-step convergence, small gradient, the iteration cap, or a
// failed line search. The final LML is never below the initial LML: if
// optimization cannot improve it, the initial parameters are restored.
func (m *Model) Train() (*OptimizeProgress, error) {
	return m.runOptimization(nil)
}

// Resume continues optimization from a checkpointed progress record.
// Completed iterations are not re-applied; optimization proceeds from the
// recorded final parameters and curvature pairs are reconstructed from
// the recorded step history.
func (m *Model) Resume(progress OptimizeProgress) (*OptimizeProgress, error) {
	return m.runOptimization(&progress)
}

func (m *Model) runOptimization(checkpoint *OptimizeProgress) (*OptimizeProgress, error) {
	optNoise := m.Options.OptimizeNoise
	nk := m.Kernel.NumParams()

	startKernel := m.Kernel.Params()
	startNoise := m.Noise
	initialParams := append([]float64(nil), startKernel...)
	if optNoise {
		initialParams = append(initialParams, startNoise)
	}

	z0 := encodeLogParams(startKernel, startNoise, optNoise)

	prog := &OptimizeProgress{
		InitialParams: append([]float64(nil), initialParams...),
	}

	startIter := 0
	if checkpoint != nil {
		if len(checkpoint.FinalParams) != nk+b2i(optNoise) {
			return nil, fmt.Errorf("gp: checkpoint parameter count %d != current %d",
				len(checkpoint.FinalParams), nk+b2i(optNoise))
		}
		if err := m.Kernel.SetParams(append([]float64(nil), checkpoint.FinalParams[:nk]...)); err != nil {
			return nil, err
		}
		if optNoise {
			m.Noise = checkpoint.FinalParams[nk]
		}
		m.invalidateCache()
		z0 = encodeLogParams(m.Kernel.Params(), m.Noise, optNoise)
		startIter = checkpoint.Iterations
		prog.InitialLML = checkpoint.InitialLML
		prog.InitialParams = append([]float64(nil), checkpoint.InitialParams...)
		prog.CompletedSteps = append([]OptimizeStep(nil), checkpoint.CompletedSteps...)
	}

	f0, g0, jit0, err := m.negLML(append([]float64(nil), z0...), optNoise)
	if err != nil {
		return nil, fmt.Errorf("gp: initial objective failed: %w", err)
	}
	if checkpoint == nil {
		prog.InitialLML = -f0
	}

	z := append([]float64(nil), z0...)
	f, g := f0, g0
	appliedJit := jit0

	sHist, yHist, rhoHist := m.seedHistory(checkpoint, optNoise)

	termination := TerminatedMaxIters
	converged := false
	stepLen := 0.0
	appliedMoves := 0
	iter := startIter
	for ; iter < startIter+m.Options.MaxIterations; iter++ {
		if infNorm(g) <= m.Options.GradientTol {
			termination = TerminatedGradient
			converged = true
			break
		}
		d := lbfgsDirection(g, sHist, yHist, rhoHist)
		newZ, newF, newG, newJit, lsStep, lsErr := backtrackingLineSearch(m, z, g, d, f, optNoise)
		if lsErr != nil {
			termination = TerminatedFailed
			break
		}
		stepLen = lsStep

		prevF := f
		s := vectorSub(newZ, z)
		y := vectorSub(newG, g)
		if sy := dot(s, y); sy > 1e-10 {
			sHist, yHist, rhoHist = pushHistory(sHist, yHist, rhoHist, s, y, 1/sy)
		}
		z, f, g, appliedJit = newZ, newF, newG, newJit
		appliedMoves++

		prog.CompletedSteps = append(prog.CompletedSteps, OptimizeStep{
			Iteration:  startIter + appliedMoves,
			Params:     expParams(z),
			NegLML:     f,
			GradNorm:   infNorm(g),
			StepLength: stepLen,
		})
		if m.Options.CheckpointHook != nil && m.Options.CheckpointEvery > 0 &&
			appliedMoves%m.Options.CheckpointEvery == 0 {
			snap := *prog
			snap.FinalLML = -f
			snap.FinalParams = expParams(z)
			snap.Iterations = startIter + appliedMoves
			snap.CompletedSteps = append([]OptimizeStep(nil), prog.CompletedSteps...)
			m.Options.CheckpointHook(&snap)
		}

		// Relative function convergence based on the accepted move.
		rel := math.Abs(prevF-f) / math.Max(1, math.Abs(f))
		if rel < m.Options.ConvergenceTol {
			termination = TerminatedConverged
			converged = true
			break
		}
	}

	// Monotonicity guarantee: never finish below the initial LML.
	if -f < prog.InitialLML {
		if err := m.applyLogParams(z0, optNoise); err != nil {
			return nil, err
		}
		_, _, appliedJit, _ = m.negLML(z0, optNoise)
		f = f0
		z = z0
	}

	prog.FinalLML = -f
	prog.FinalParams = expParams(z)
	prog.Iterations = startIter + appliedMoves
	prog.Converged = converged
	prog.Termination = termination
	prog.AppliedJitter = appliedJit
	cp := *prog
	m.progress = &cp
	return prog, nil
}

// seedHistory reconstructs L-BFGS curvature pairs from checkpoint steps.
// Parameter vectors are stored in natural space; gradients are recomputed.
func (m *Model) seedHistory(cp *OptimizeProgress, optNoise bool) (s, y [][]float64, rho []float64) {
	if cp == nil || len(cp.CompletedSteps) < 2 {
		return nil, nil, nil
	}
	steps := cp.CompletedSteps
	lo := 0
	if len(steps) > lbfgsMem+1 {
		lo = len(steps) - lbfgsMem - 1
	}
	var prevZ, prevG []float64
	for i := lo; i < len(steps); i++ {
		zz := logParams(steps[i].Params)
		_, gg, _, err := m.negLML(zz, optNoise)
		if err != nil {
			return nil, nil, nil
		}
		if prevZ != nil {
			si := vectorSub(zz, prevZ)
			yi := vectorSub(gg, prevG)
			if syv := dot(si, yi); syv > 1e-10 {
				s, y, rho = pushHistory(s, y, rho, si, yi, 1/syv)
			}
		}
		prevZ, prevG = zz, gg
	}
	return s, y, rho
}

// lbfgsDirection applies the two-loop recursion, returning -H g.
func lbfgsDirection(g []float64, s, y [][]float64, rho []float64) []float64 {
	q := append([]float64(nil), g...)
	nHist := len(s)
	alpha := make([]float64, nHist)
	for i := nHist - 1; i >= 0; i-- {
		alpha[i] = rho[i] * dot(s[i], q)
		q = vectorAddScaled(q, y[i], -alpha[i])
	}
	gamma := 1.0
	if nHist > 0 {
		gamma = dot(s[nHist-1], y[nHist-1]) / math.Max(dot(y[nHist-1], y[nHist-1]), 1e-300)
	}
	r := scale(q, gamma)
	for i := 0; i < nHist; i++ {
		beta := rho[i] * dot(y[i], r)
		r = vectorAddScaled(r, s[i], alpha[i]-beta)
	}
	for i := range r {
		r[i] = -r[i]
	}
	return r
}

// backtrackingLineSearch enforces Armijo sufficient decrease.
func backtrackingLineSearch(m *Model, z, g, d []float64, f float64, optNoise bool) (nz []float64, nf float64, ng []float64, jit, step float64, err error) {
	const (
		c1       = 1e-4
		shrink   = 0.5
		maxTries = 40
	)
	dirSlope := dot(g, d)
	if dirSlope >= 0 {
		d = scale(g, -1)
		dirSlope = dot(g, d)
	}
	t := 1.0
	for tries := 0; tries < maxTries; tries++ {
		cand := make([]float64, len(z))
		for i := range z {
			cand[i] = clamp(z[i]+t*d[i], logParamLower, logParamUpper)
		}
		cf, cg, cjit, ferr := m.negLML(cand, optNoise)
		if ferr == nil && !math.IsNaN(cf) && !math.IsInf(cf, 0) &&
			cf <= f+c1*t*dirSlope+1e-12 {
			return cand, cf, cg, cjit, t, nil
		}
		t *= shrink
	}
	return nil, 0, nil, 0, 0, fmt.Errorf("gp: line search failed to find sufficient decrease")
}

func pushHistory(s, y [][]float64, rho, sv, yv []float64, rhov float64) ([][]float64, [][]float64, []float64) {
	s = append(s, sv)
	y = append(y, yv)
	rho = append(rho, rhov)
	if len(s) > lbfgsMem {
		s = s[1:]
		y = y[1:]
		rho = rho[1:]
	}
	return s, y, rho
}

func encodeLogParams(kernelParams []float64, noise float64, optimizeNoise bool) []float64 {
	z := make([]float64, len(kernelParams)+b2i(optimizeNoise))
	for i, v := range kernelParams {
		z[i] = math.Log(v)
	}
	if optimizeNoise {
		z[len(kernelParams)] = math.Log(math.Max(noise, math.Exp(logParamLower)))
	}
	return z
}

func expParams(z []float64) []float64 {
	out := make([]float64, len(z))
	for i, v := range z {
		out[i] = math.Exp(v)
	}
	return out
}

func logParams(natural []float64) []float64 {
	z := make([]float64, len(natural))
	for i, v := range natural {
		z[i] = math.Log(math.Max(v, math.Exp(logParamLower)))
	}
	return z
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func infNorm(a []float64) float64 {
	mx := 0.0
	for _, v := range a {
		if av := math.Abs(v); av > mx {
			mx = av
		}
	}
	return mx
}

func vectorSub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func vectorAddScaled(a, b []float64, s float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + s*b[i]
	}
	return out
}

func scale(a []float64, s float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * s
	}
	return out
}
