package gpr

import (
	"errors"
	"math"
)

// KernelType 标识核函数种类。
type KernelType string

const (
	KernelRBF    KernelType = "rbf"
	KernelMatern KernelType = "matern32"
	KernelLinear KernelType = "linear"
	KernelRQ     KernelType = "rational_quadratic"
)

// ErrUnknownKernel 表示配置了不支持的核函数。
var ErrUnknownKernel = errors.New("gpr: unknown kernel type")

// KernelParams 是核函数的超参数（均为正数）。
// LengthScale 为特征长度尺度；SignalVariance 为信号方差；
// Alpha 仅 RationalQuadratic 使用（相对尺度混合参数）。
type KernelParams struct {
	LengthScale    float64
	SignalVariance float64
	Alpha          float64
}

// Kernel 表示一个可计算两点协方差的核函数。
type Kernel interface {
	Type() KernelType
	Params() KernelParams
	// Value 返回 k(a, b)。
	Value(a, b []float64) float64
}

// baseKernel 保存四种核共用的参数。
type baseKernel struct {
	kind   KernelType
	params KernelParams
}

func (k baseKernel) Type() KernelType     { return k.kind }
func (k baseKernel) Params() KernelParams { return k.params }

// rbfKernel 为平方指数（RBF）核：sf² exp(-r²/(2ℓ²))。
type rbfKernel struct{ baseKernel }

// matern32Kernel 为 Matérn 3/2 核：sf² (1+√3 r/ℓ) exp(-√3 r/ℓ)。
type matern32Kernel struct{ baseKernel }

// linearKernel 为归一化线性核：sf² a·b/ℓ²。
type linearKernel struct{ baseKernel }

// rationalQuadraticKernel：sf² (1 + r²/(2αℓ²))^(-α)。
type rationalQuadraticKernel struct{ baseKernel }

func (k *rbfKernel) Value(a, b []float64) float64 {
	r2 := distanceSquared(a, b)
	return k.params.SignalVariance * math.Exp(-r2/(2.0*k.params.LengthScale*k.params.LengthScale))
}

func (k *matern32Kernel) Value(a, b []float64) float64 {
	r := math.Sqrt(distanceSquared(a, b))
	z := math.Sqrt(3.0) * r / k.params.LengthScale
	return k.params.SignalVariance * (1.0 + z) * math.Exp(-z)
}

func (k *linearKernel) Value(a, b []float64) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return k.params.SignalVariance * dot / (k.params.LengthScale * k.params.LengthScale)
}

func (k *rationalQuadraticKernel) Value(a, b []float64) float64 {
	r2 := distanceSquared(a, b)
	inv := 1.0 + r2/(2.0*k.params.Alpha*k.params.LengthScale*k.params.LengthScale)
	return k.params.SignalVariance * math.Pow(inv, -k.params.Alpha)
}

// NewKernel 按类型与参数构造核函数。
func NewKernel(t KernelType, p KernelParams) (Kernel, error) {
	if p.LengthScale <= 0 || math.IsNaN(p.LengthScale) || math.IsInf(p.LengthScale, 0) {
		return nil, errors.New("gpr: length scale must be a positive finite number")
	}
	if p.SignalVariance <= 0 || math.IsNaN(p.SignalVariance) || math.IsInf(p.SignalVariance, 0) {
		return nil, errors.New("gpr: signal variance must be a positive finite number")
	}
	base := baseKernel{kind: t, params: p}
	switch t {
	case KernelRBF:
		return &rbfKernel{base}, nil
	case KernelMatern:
		return &matern32Kernel{base}, nil
	case KernelLinear:
		return &linearKernel{base}, nil
	case KernelRQ:
		if p.Alpha <= 0 || math.IsNaN(p.Alpha) || math.IsInf(p.Alpha, 0) {
			return nil, errors.New("gpr: alpha must be a positive finite number")
		}
		return &rationalQuadraticKernel{base}, nil
	default:
		return nil, ErrUnknownKernel
	}
}

// distanceSquared 返回两点欧氏距离平方。
func distanceSquared(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
