package gpr

import (
	"math"
	"time"
)

// StopReason 说明优化终止原因。
type StopReason string

const (
	StopConverged   StopReason = "converged"
	StopMaxIters    StopReason = "max_iterations"
	StopNoImprove   StopReason = "no_improvement"
	StopFailed      StopReason = "numeric_failure"
	StopNotOptimize StopReason = "optimization_disabled"
)

// OptimizerConfig 配置基于 Adam 的 LML 最大化。
type OptimizerConfig struct {
	MaxIterations   int
	Tolerance       float64
	LearningRate    float64
	GradientEpsilon float64
	NoImproveWindow int
	Seed            int64
	CheckpointEvery int
}

// defaultOptimizerConfig 返回带默认值的优化器配置。
func defaultOptimizerConfig() OptimizerConfig {
	return OptimizerConfig{
		MaxIterations:   200,
		Tolerance:       1e-6,
		LearningRate:    0.05,
		GradientEpsilon: 1e-5,
		NoImproveWindow: 10,
		Seed:            1,
		CheckpointEvery: 10,
	}
}

// OptimizeResult 记录一次完整（或恢复后完成的）优化的结果。
type OptimizeResult struct {
	Kernel     Kernel
	Noise      float64
	InitialLML float64
	FinalLML   float64
	Iterations int
	StopReason StopReason
	History    []IterationStat
	Elapsed    time.Duration
	JitterUsed float64
}

// IterationStat 是单个迭代的进度记录。
type IterationStat struct {
	Iteration int
	LML       float64
	Params    KernelParams
	Noise     float64
}

// optimizerState 是可序列化的优化器内部状态，用于断点续训。
type optimizerState struct {
	Iteration int
	LML       float64
	Params    []float64
	M         []float64
	V         []float64
}

// packParams 将核参数与噪声打包为对数空间向量。
func packParams(k Kernel, noise float64) []float64 {
	p := k.Params()
	if k.Type() == KernelRQ {
		return []float64{
			math.Log(p.LengthScale),
			math.Log(p.SignalVariance),
			math.Log(p.Alpha),
			math.Log(math.Max(noise, minNoiseStd)),
		}
	}
	return []float64{
		math.Log(p.LengthScale),
		math.Log(p.SignalVariance),
		math.Log(math.Max(noise, minNoiseStd)),
	}
}

// unpackParams 将对数空间向量还原为核参数与噪声。
func unpackParams(t KernelType, packed []float64) (KernelParams, float64, error) {
	want := paramDim(t)
	if len(packed) != want {
		return KernelParams{}, 0, ErrDimensionMismatch
	}
	p := KernelParams{
		LengthScale:    math.Exp(clampPacked(packed[0])),
		SignalVariance: math.Exp(clampPacked(packed[1])),
		Alpha:          1,
	}
	noise := math.Exp(clampPacked(packed[want-1]))
	if t == KernelRQ {
		p.Alpha = math.Exp(clampPacked(packed[2]))
	}
	return p, noise, nil
}

// evaluateLML 计算给定参数下的对数边际似然。
func evaluateLML(data *Dataset, t KernelType, packed []float64, sc StabilityConfig) (float64, []float64, error) {
	params, noise, err := unpackParams(t, packed)
	if err != nil {
		return math.NaN(), nil, err
	}
	k, err := NewKernel(t, params)
	if err != nil {
		return math.NaN(), nil, err
	}
	n := data.Len()
	ky := kernelMatrix(data.X, k)
	noiseVar := noise * noise
	for i := 0; i < n; i++ {
		ky[i][i] += noiseVar
	}
	meanTrain := meanValue(data.Y)
	centered := make([]float64, n)
	for i := range centered {
		centered[i] = data.Y[i] - meanTrain
	}
	// 一次性分解，供二次型与对数行列式复用。
	l, err := choleskyFactor(ky, sc)
	if err != nil {
		return math.NaN(), nil, err
	}
	yVec := forwardSubstitution(l, centered)
	alpha := backSubstitution(l, yVec)
	var quad float64
	for i := range centered {
		quad += centered[i] * alpha[i]
	}
	lml := -0.5*quad - logDetCholesky(l) - 0.5*float64(n)*math.Log(2*math.Pi)
	if math.IsNaN(lml) || math.IsInf(lml, 0) {
		return math.NaN(), nil, ErrSingular
	}
	return lml, nil, nil
}

// runOptimizer 执行（或从 state 恢复执行）Adam 优化。
// state 为 nil 表示从头开始；否则从 state.Iteration+1 继续。
func runOptimizer(data *Dataset, cfg TrainConfig, state *optimizerState) (*OptimizeResult, *optimizerState, error) {
	start := time.Now()
	oc := cfg.Optimizer
	sc := cfg.Stability
	dim := paramDim(cfg.KernelType)

	initial := packParams(mustKernel(cfg.KernelType, cfg.InitialParams), cfg.Noise)
	initialLML, _, err := evaluateLML(data, cfg.KernelType, initial, sc)
	if err != nil {
		return nil, nil, err
	}

	result := &OptimizeResult{
		Noise:      cfg.Noise,
		InitialLML: initialLML,
		FinalLML:   initialLML,
		StopReason: StopNotOptimize,
		History:    []IterationStat{},
	}
	bestPacked := append([]float64(nil), initial...)

	if !cfg.Optimize {
		result.Kernel, _ = NewKernel(cfg.KernelType, cfg.InitialParams)
		result.Elapsed = time.Since(start)
		return result, nil, nil
	}

	params := append([]float64(nil), initial...)
	mom := make([]float64, dim)
	vel := make([]float64, dim)
	iter := 0
	bestLML := initialLML
	lastImprove := 0

	if state != nil {
		if len(state.Params) != dim {
			return nil, nil, ErrCheckpointCorrupt
		}
		params = append([]float64(nil), state.Params...)
		mom = append([]float64(nil), state.M...)
		vel = append([]float64(nil), state.V...)
		iter = state.Iteration
		bestLML = state.LML
		result.FinalLML = bestLML
		bestPacked = append([]float64(nil), params...)
		// 无法精确恢复距上次改进的迭代数时，用当前最佳点保守初始化，
		// 避免续训后立即错误触发 no_improvement。
		lastImprove = iter
		result.History = []IterationStat{statFromPacked(iter, bestLML, cfg.KernelType, params)}
	}

	stop := StopMaxIters
	for iter < oc.MaxIterations {
		lml, grad, err := lmlWithGradient(data, cfg.KernelType, params, sc, oc.GradientEpsilon)
		if err != nil {
			stop = StopFailed
			break
		}
		// 若新参数使 LML 下降，Adam 的动量更新仍继续，但 best 只记录改进。
		if lml > bestLML+1e-12 {
			bestLML = lml
			bestPacked = append([]float64(nil), params...)
			lastImprove = iter
		}
		result.History = append(result.History, statFromPacked(iter+1, lml, cfg.KernelType, params))

		// 梯度上升方向。
		beta1, beta2, eps := 0.9, 0.999, 1e-8
		gmax := 0.0
		for i := 0; i < dim; i++ {
			mom[i] = beta1*mom[i] + (1-beta1)*grad[i]
			vel[i] = beta2*vel[i] + (1-beta2)*grad[i]*grad[i]
			mHat := mom[i] / (1 - math.Pow(beta1, float64(iter+1)))
			vHat := vel[i] / (1 - math.Pow(beta2, float64(iter+1)))
			step := oc.LearningRate * mHat / (math.Sqrt(vHat) + eps)
			params[i] += step
			params[i] = clampPacked(params[i])
			if a := math.Abs(step); a > gmax {
				gmax = a
			}
		}
		iter++

		if gmax < oc.Tolerance {
			stop = StopConverged
			break
		}
		if iter-lastImprove >= oc.NoImproveWindow && iter >= oc.NoImproveWindow {
			stop = StopNoImprove
			break
		}

		// 中间状态通过 checkpoint 回调由上层落盘（见 service 续训）。
		if oc.CheckpointEvery > 0 && iter%oc.CheckpointEvery == 0 {
			state = &optimizerState{
				Iteration: iter,
				LML:       bestLML,
				Params:    append([]float64(nil), params...),
				M:         append([]float64(nil), mom...),
				V:         append([]float64(nil), vel...),
			}
			if cfg.onCheckpoint != nil {
				if err := cfg.onCheckpoint(iter, bestLML, params, mom, vel); err != nil {
					return nil, state, err
				}
			}
		}
	}

	// 保证优化后 LML 不低于初始值：最终参数取评估过的最佳点。
	finalLML, _, err := evaluateLML(data, cfg.KernelType, bestPacked, sc)
	if err != nil || finalLML < initialLML-1e-8 {
		bestPacked = initial
		finalLML = initialLML
		stop = StopNoImprove
	}
	bestParams, bestNoise, err := unpackParams(cfg.KernelType, bestPacked)
	if err != nil {
		return nil, nil, err
	}
	result.Kernel, _ = NewKernel(cfg.KernelType, bestParams)
	result.Noise = bestNoise
	result.FinalLML = finalLML
	result.Iterations = iter
	result.StopReason = stop
	result.Elapsed = time.Since(start)
	return result, nil, nil
}

const (
	minNoiseStd = 1e-8
	maxLogParam = 8.0
	minLogParam = -8.0
)

func paramDim(t KernelType) int {
	if t == KernelRQ {
		return 4
	}
	return 3
}

func clampPacked(v float64) float64 {
	if math.IsNaN(v) {
		return minLogParam
	}
	if v < minLogParam {
		return minLogParam
	}
	if v > maxLogParam {
		return maxLogParam
	}
	return v
}

func mustKernel(t KernelType, p KernelParams) Kernel {
	k, err := NewKernel(t, p)
	if err != nil {
		panic(err)
	}
	return k
}

func meanValue(y []float64) float64 {
	var sum float64
	for _, v := range y {
		sum += v
	}
	return sum / float64(len(y))
}

// lmlWithGradient 用中心差分计算 LML 对打包参数的数值梯度。
func lmlWithGradient(data *Dataset, t KernelType, params []float64, sc StabilityConfig, eps float64) (float64, []float64, error) {
	lml, _, err := evaluateLML(data, t, params, sc)
	if err != nil {
		return math.NaN(), nil, err
	}
	grad := make([]float64, len(params))
	for i := range params {
		up := append([]float64(nil), params...)
		dn := append([]float64(nil), params...)
		up[i] += eps
		dn[i] -= eps
		lUp, _, errUp := evaluateLML(data, t, up, sc)
		lDn, _, errDn := evaluateLML(data, t, dn, sc)
		if errUp != nil || errDn != nil {
			return math.NaN(), nil, ErrSingular
		}
		grad[i] = (lUp - lDn) / (2 * eps)
	}
	return lml, grad, nil
}

func statFromPacked(iter int, lml float64, t KernelType, packed []float64) IterationStat {
	p, noise, _ := unpackParams(t, packed)
	return IterationStat{Iteration: iter, LML: lml, Params: p, Noise: noise}
}
