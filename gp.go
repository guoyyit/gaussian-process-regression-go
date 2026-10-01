package gpr

import (
	"errors"
	"math"
)

// errInvalidNoise 表示噪声参数非法（负数或非有限值）。
var errInvalidNoise = errors.New("gpr: noise standard deviation must be a non-negative finite number")

// Model 是一个训练完成、不可变的 GP 模型快照。
type Model struct {
	data      *Dataset
	kernel    Kernel
	noise     float64
	alpha     []float64   // (K + σ²I)⁻¹ (y - m)
	factor    [][]float64 // L：K + σ²I = L Lᵀ
	meanTrain float64
	stability StabilityConfig
}

// TrainConfig 控制一次训练（拟合）过程。
type TrainConfig struct {
	KernelType    KernelType
	InitialParams KernelParams
	Noise         float64
	Optimize      bool
	Stability     StabilityConfig
	Optimizer     OptimizerConfig
	// onCheckpoint 由 Service 注入，用于在优化迭代中途原子落盘（内部字段）。
	onCheckpoint func(iteration int, lml float64, params, mom, vel []float64) error
}

// StabilityConfig 配置数值稳定性策略。
type StabilityConfig struct {
	Policy        jitterPolicy
	InitialJitter float64
	MaxJitter     float64
	MinVariance   float64
}

// Prediction 是单点预测结果。
type Prediction struct {
	Mean     float64
	Variance float64
}

// defaultTrainConfig 返回带默认值的训练配置。
func defaultTrainConfig() TrainConfig {
	return TrainConfig{
		KernelType:    KernelRBF,
		InitialParams: KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1},
		Noise:         1e-2,
		Optimize:      true,
		Stability:     DefaultStabilityConfig(),
		Optimizer:     defaultOptimizerConfig(),
	}
}

// DefaultStabilityConfig 返回推荐的数值稳定性配置（自适应抖动）。
func DefaultStabilityConfig() StabilityConfig {
	return StabilityConfig{
		Policy:        JitterAdaptive,
		InitialJitter: 1e-10,
		MaxJitter:     1e-2,
		MinVariance:   0,
	}
}

// validateTrainConfig 校验训练配置并补全默认值。
func validateTrainConfig(cfg TrainConfig, dim int) (TrainConfig, error) {
	def := defaultTrainConfig()
	if cfg.KernelType == "" {
		cfg.KernelType = def.KernelType
	}
	if cfg.InitialParams.LengthScale == 0 {
		cfg.InitialParams.LengthScale = def.InitialParams.LengthScale
	}
	if cfg.InitialParams.SignalVariance == 0 {
		cfg.InitialParams.SignalVariance = def.InitialParams.SignalVariance
	}
	if cfg.InitialParams.Alpha == 0 {
		cfg.InitialParams.Alpha = def.InitialParams.Alpha
	}
	if cfg.Noise == 0 {
		cfg.Noise = def.Noise
	}
	if cfg.Noise < 0 || math.IsNaN(cfg.Noise) || math.IsInf(cfg.Noise, 0) {
		return cfg, errInvalidNoise
	}
	if cfg.Stability == (StabilityConfig{}) {
		cfg.Stability = def.Stability
	}
	if cfg.Stability.InitialJitter <= 0 {
		cfg.Stability.InitialJitter = def.Stability.InitialJitter
	}
	if cfg.Optimizer == (OptimizerConfig{}) {
		cfg.Optimizer = def.Optimizer
	}
	if _, err := NewKernel(cfg.KernelType, cfg.InitialParams); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// fitGP 用给定数据与参数训练一个不可变模型（不做超参优化）。
func fitGP(data *Dataset, k Kernel, noise float64, meanTrain float64, sc StabilityConfig) (*Model, error) {
	n := data.Len()
	ky := kernelMatrix(data.X, k)
	noiseVar := noise * noise
	for i := 0; i < n; i++ {
		ky[i][i] += noiseVar
	}
	centered := make([]float64, n)
	for i := 0; i < n; i++ {
		centered[i] = data.Y[i] - meanTrain
	}
	l, err := choleskyFactor(ky, sc)
	if err != nil {
		return nil, err
	}
	alpha := backSubstitution(l, forwardSubstitution(l, centered))
	return &Model{
		data:      data,
		kernel:    k,
		noise:     noise,
		alpha:     alpha,
		factor:    l,
		meanTrain: meanTrain,
		stability: sc,
	}, nil
}

// Predict 返回单点的均值与后验方差。
func (m *Model) Predict(x []float64) (Prediction, error) {
	if len(x) != m.data.Dim() {
		return Prediction{}, ErrDimensionMismatch
	}
	n := m.data.Len()
	kStar := make([]float64, n)
	var mean float64
	for i := 0; i < n; i++ {
		kStar[i] = m.kernel.Value(m.data.Row(i), x)
		mean += kStar[i] * m.alpha[i]
	}
	mean += m.meanTrain

	v := backSubstitution(m.factor, forwardSubstitution(m.factor, kStar))
	var kss float64
	for i := 0; i < n; i++ {
		kss += kStar[i] * v[i]
	}
	variance := m.kernel.Value(x, x) - kss
	minVar := m.stability.MinVariance
	if variance < minVar {
		variance = minVar
	}
	if variance < 0 && variance > -1e-8 {
		variance = 0
	}
	if variance < 0 || math.IsNaN(variance) || math.IsInf(variance, 0) ||
		math.IsNaN(mean) || math.IsInf(mean, 0) {
		return Prediction{}, ErrSingular
	}
	return Prediction{Mean: mean, Variance: variance}, nil
}

// PredictBatch 批量预测。
func (m *Model) PredictBatch(x [][]float64) ([]Prediction, error) {
	if _, err := checkPoints(x, m.data.Dim()); err != nil {
		return nil, err
	}
	out := make([]Prediction, len(x))
	for i := range x {
		p, err := m.Predict(x[i])
		if err != nil {
			return nil, err
		}
		out[i] = p
	}
	return out, nil
}

// LogMarginalLikelihood 返回当前超参数下的对数边际似然及其梯度（对打包参数）。
func (m *Model) LogMarginalLikelihood() (float64, []float64, error) {
	return evaluateLML(m.data, m.kernel.Type(), packParams(m.kernel, m.noise), m.stability)
}

// kernelMatrix 构造 n×n 核矩阵 K_ij = k(x_i, x_j)。
func kernelMatrix(x [][]float64, kern Kernel) [][]float64 {
	n := len(x)
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := kern.Value(x[i], x[j])
			m[i][j] = v
			m[j][i] = v
		}
	}
	return m
}
