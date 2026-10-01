package gpr

import (
	"errors"
	"math"
)

// ErrEmptyData 表示训练数据为空。
var ErrEmptyData = errors.New("gpr: training data is empty")

// ErrDimensionMismatch 表示输入维度与目标数量不一致。
var ErrDimensionMismatch = errors.New("gpr: input dimension mismatch")

// Dataset 表示一组带噪声的观测。
// X 的每行是一个输入点，长度为 D；Y 为对应的标量观测，长度为 N。
type Dataset struct {
	X [][]float64
	Y []float64
}

// NewDataset 校验并构造数据集。
func NewDataset(x [][]float64, y []float64) (*Dataset, error) {
	if len(x) == 0 {
		return nil, ErrEmptyData
	}
	dim, err := checkPoints(x, 0)
	if err != nil {
		return nil, err
	}
	if len(y) != len(x) {
		return nil, ErrDimensionMismatch
	}
	for _, v := range y {
		if !isFinite(v) {
			return nil, errors.New("gpr: target contains NaN or Inf")
		}
	}
	cp := make([][]float64, len(x))
	for i := range x {
		row := make([]float64, dim)
		copy(row, x[i])
		cp[i] = row
	}
	yCopy := make([]float64, len(y))
	copy(yCopy, y)
	return &Dataset{X: cp, Y: yCopy}, nil
}

// Dim 返回输入维度，空数据集返回 0。
func (d *Dataset) Dim() int {
	if d == nil || len(d.X) == 0 {
		return 0
	}
	return len(d.X[0])
}

// Len 返回观测点数量。
func (d *Dataset) Len() int {
	if d == nil {
		return 0
	}
	return len(d.X)
}

// Append 追加观测点并返回新的数据集。
func (d *Dataset) Append(x [][]float64, y []float64) (*Dataset, error) {
	if len(x) != len(y) {
		return nil, ErrDimensionMismatch
	}
	if len(x) == 0 {
		return d, nil
	}
	if _, err := checkPoints(x, d.Dim()); err != nil {
		return nil, err
	}
	added, err := NewDataset(x, y)
	if err != nil {
		return nil, err
	}
	next := &Dataset{
		X: make([][]float64, 0, d.Len()+len(x)),
		Y: make([]float64, 0, d.Len()+len(y)),
	}
	next.X = append(next.X, d.X...)
	next.Y = append(next.Y, d.Y...)
	next.X = append(next.X, added.X...)
	next.Y = append(next.Y, added.Y...)
	return next, nil
}

// Row 以 []float64 返回第 i 个输入点。
func (d *Dataset) Row(i int) []float64 {
	return d.X[i]
}

// checkPoints 校验一批输入点的维度是否一致且等于 wantDim（wantDim<=0 时不约束）。
func checkPoints(points [][]float64, wantDim int) (int, error) {
	if len(points) == 0 {
		return 0, ErrEmptyData
	}
	dim := len(points[0])
	if dim == 0 {
		return 0, errors.New("gpr: input points must have at least one dimension")
	}
	if wantDim > 0 && dim != wantDim {
		return 0, ErrDimensionMismatch
	}
	for _, p := range points {
		if len(p) != dim {
			return 0, ErrDimensionMismatch
		}
		for _, v := range p {
			if !isFinite(v) {
				return 0, errors.New("gpr: input contains NaN or Inf")
			}
		}
	}
	return dim, nil
}

// isFinite 判断 f 是否既非 NaN 也非无穷。
func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}
