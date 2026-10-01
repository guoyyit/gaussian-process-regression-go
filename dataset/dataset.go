package dataset

import (
	"math"
	"math/rand"
)

// Generator produces reproducible noisy observations.
//
// Determinism: a Generator holds a math/rand.Rand constructed from a fixed
// seed. Two generators with the same seed, target function, dimension and
// Sample parameters produce byte-identical datasets, even across process
// restarts (the global rand source is never used).
type Generator struct {
	rng *rand.Rand
	fn  func(x []float64) float64
	dim int
}

// New creates a generator sampling inputs uniformly from [-1, 1]^dim.
// A nil fn defaults to Sinusoid1D (dimension 1); higher dimensions must
// supply their own target function.
func New(seed int64, dim int, fn func(x []float64) float64) *Generator {
	if fn == nil {
		if dim == 1 {
			fn = Sinusoid1D
		} else {
			fn = func(x []float64) float64 {
				s := 0.0
				for _, v := range x {
					s += v * v
				}
				return s
			}
		}
	}
	return &Generator{rng: rand.New(rand.NewSource(seed)), fn: fn, dim: dim}
}

// Sample returns n observations y = fn(x) + eps with eps ~ N(0, noiseStd^2).
func (g *Generator) Sample(n int, noiseStd float64) (x [][]float64, y []float64) {
	x = make([][]float64, n)
	y = make([]float64, n)
	for i := 0; i < n; i++ {
		xi := make([]float64, g.dim)
		for d := 0; d < g.dim; d++ {
			xi[d] = g.rng.Float64()*2 - 1
		}
		x[i] = xi
		y[i] = g.fn(xi) + g.rng.NormFloat64()*noiseStd
	}
	return x, y
}

// Sinc1D is the normalized sinc target sin(x)/x (with value 1 at 0).
func Sinc1D(x []float64) float64 {
	v := x[0]
	if math.Abs(v) < 1e-12 {
		return 1
	}
	return math.Sin(v) / v
}

// Sinusoid1D is a smooth sin target.
func Sinusoid1D(x []float64) float64 {
	return math.Sin(2 * x[0])
}

// Grid1D builds n evenly spaced 1D inputs over [lo, hi].
func Grid1D(lo, hi float64, n int) [][]float64 {
	if n <= 0 {
		return nil
	}
	out := make([][]float64, n)
	if n == 1 {
		out[0] = []float64{lo}
		return out
	}
	step := (hi - lo) / float64(n-1)
	for i := 0; i < n; i++ {
		out[i] = []float64{lo + step*float64(i)}
	}
	return out
}

// NoisyGrid evaluates fn on Grid1D points and adds reproducible noise.
func NoisyGrid(seed int64, lo, hi float64, n int, noiseStd float64, fn func(x []float64) float64) ([][]float64, []float64) {
	grid := Grid1D(lo, hi, n)
	rng := rand.New(rand.NewSource(seed))
	y := make([]float64, n)
	for i, xi := range grid {
		y[i] = fn(xi) + rng.NormFloat64()*noiseStd
	}
	return grid, y
}
