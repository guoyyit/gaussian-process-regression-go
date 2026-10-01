package gpr

import (
	"errors"
	"math"
)

// ErrSingular 表示矩阵病态到无法获得可靠解。
var ErrSingular = errors.New("gpr: matrix is singular or ill-conditioned")

// jitterPolicy 描述核矩阵病态时的稳定性策略。
type jitterPolicy int

const (
	// JitterOff 不添加抖动；病态时直接返回 ErrSingular。
	JitterOff jitterPolicy = iota
	// JitterFixed 仅添加一次固定抖动。
	JitterFixed
	// JitterAdaptive 按倍增序列重试直到求解成功或达到上限。
	JitterAdaptive
)

// cloneMatrix 返回矩阵的深拷贝。
func cloneMatrix(a [][]float64) [][]float64 {
	n := len(a)
	c := make([][]float64, n)
	for i := range a {
		c[i] = make([]float64, len(a[i]))
		copy(c[i], a[i])
	}
	return c
}

// choleskySolve 求解 A x = b，其中 A 为对称正定矩阵。
// strategy 控制病态时的抖动重试行为，jitter 为初始抖动量级。
func choleskySolve(a [][]float64, b []float64, strategy jitterPolicy, jitter float64) ([]float64, error) {
	if len(b) != len(a) {
		return nil, ErrSingular
	}
	sc := StabilityConfig{Policy: strategy, InitialJitter: jitter, MaxJitter: 1e-2}
	l, err := choleskyFactor(a, sc)
	if err != nil {
		return nil, err
	}
	y := forwardSubstitution(l, b)
	x := backSubstitution(l, y)
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, ErrSingular
		}
	}
	return x, nil
}

// choleskyFactor 返回对称化后的 A 的 Cholesky 下三角因子 L（A = L Lᵀ）。
// 不做抖动重试；调用方通过 StabilityConfig.Policy 在外部循环重试。
func choleskyFactor(a [][]float64, sc StabilityConfig) ([][]float64, error) {
	n := len(a)
	if n == 0 {
		return nil, ErrSingular
	}
	m := cloneMatrix(a)
	// 对称化，消除调用方可能产生的不对称舍入误差。
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := 0.5 * (m[i][j] + m[j][i])
			m[i][j] = avg
			m[j][i] = avg
		}
	}
	var diagScale float64
	for i := 0; i < n; i++ {
		diagScale = math.Max(diagScale, math.Abs(m[i][i]))
	}
	if diagScale == 0 || math.IsNaN(diagScale) || math.IsInf(diagScale, 1) {
		return nil, ErrSingular
	}
	tol := 1e-10 * diagScale

	// 自适应抖动：在分解内部逐级加大对角元，避免上层重复构造矩阵。
	jitter := sc.InitialJitter
	if sc.Policy == JitterOff {
		jitter = 0
	}
	if sc.Policy == JitterAdaptive && sc.InitialJitter <= 0 {
		jitter = 1e-10
	}
	maxJitter := sc.MaxJitter
	if maxJitter <= 0 {
		maxJitter = 1e-2
	}

	for {
		mm := cloneMatrix(m)
		if jitter > 0 {
			for i := 0; i < n; i++ {
				mm[i][i] += jitter
			}
		}
		l := make([][]float64, n)
		for i := range l {
			l[i] = make([]float64, n)
		}
		failed := false
		for i := 0; i < n && !failed; i++ {
			for j := 0; j <= i; j++ {
				var sum float64
				for kk := 0; kk < j; kk++ {
					sum += l[i][kk] * l[j][kk]
				}
				if i == j {
					val := mm[i][i] - sum
					if !(val > tol) {
						failed = true
						break
					}
					l[i][j] = math.Sqrt(val)
				} else {
					l[i][j] = (mm[i][j] - sum) / l[j][j]
				}
			}
		}
		if !failed {
			return l, nil
		}
		switch sc.Policy {
		case JitterOff:
			return nil, ErrSingular
		case JitterFixed:
			if jitter > 0 {
				return nil, ErrSingular
			}
			jitter = sc.InitialJitter
			if jitter <= 0 {
				return nil, ErrSingular
			}
		case JitterAdaptive:
			jitter *= 10
			if jitter > maxJitter {
				return nil, ErrSingular
			}
		}
	}
}

// logDetCholesky 返回 Cholesky 因子对角元对数之和（即 0.5*log|A|）。
func logDetCholesky(l [][]float64) float64 {
	var sum float64
	for i := range l {
		sum += math.Log(l[i][i])
	}
	return sum
}

// forwardSubstitution 求解 L y = b（L 下三角）。
func forwardSubstitution(l [][]float64, b []float64) []float64 {
	n := len(l)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j < i; j++ {
			sum += l[i][j] * y[j]
		}
		y[i] = (b[i] - sum) / l[i][i]
	}
	return y
}

// backSubstitution 求解 Lᵀ x = b（L 下三角）。
func backSubstitution(l [][]float64, b []float64) []float64 {
	n := len(l)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		var sum float64
		for j := i + 1; j < n; j++ {
			sum += l[j][i] * x[j]
		}
		x[i] = (b[i] - sum) / l[i][i]
	}
	return x
}
