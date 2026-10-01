package gp

import (
	"math"
	"testing"

	"github.com/guoyyit/gaussian-process-regression-go/kernel"
)

func TestARDMultiDimensionOptimization(t *testing.T) {
	n := 30
	x := make([][]float64, n)
	y := make([]float64, n)
	// f(x0, x1) depends mostly on x0.
	for i := range x {
		u0 := -2 + 4*float64(i%10)/9
		u1 := -2 + 4*float64(i/10)/2
		x[i] = []float64{u0, u1}
		y[i] = math.Sin(u0) + 0.05*u1
	}
	k, err := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 2, LengthScales: []float64{1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultOptions()
	opts.NoiseVariance = 0.02
	opts.OptimizeNoise = true
	opts.MaxIterations = 60
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	prog, err := m.Train()
	if err != nil {
		t.Fatal(err)
	}
	if prog.FinalLML < prog.InitialLML {
		t.Fatal("ARD optimization reduced LML")
	}
	p, err := m.Predict([]float64{0.5, 0.1})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || p.Variance < 0 {
		t.Fatalf("bad ARD prediction %+v", p)
	}
}

func TestStrictModeLMLReturnsError(t *testing.T) {
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{1}})
	x := [][]float64{{0}, {0}, {0}}
	y := []float64{1, 1, 1}
	opts := DefaultOptions()
	opts.NoiseVariance = 1e-300
	opts.Stability = StabilityStrict
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.LogMarginalLikelihood(); err == nil {
		t.Fatal("expected LML to fail in strict mode for singular matrix")
	}
}

func TestPredictionVarianceNeverNegative(t *testing.T) {
	m := train1D(t, 0.01, StabilityJitter)
	for i := -20; i <= 20; i++ {
		p, err := m.Predict([]float64{float64(i) * 0.3})
		if err != nil {
			t.Fatal(err)
		}
		if p.Variance < 0 || p.ObservedVariance < 0 {
			t.Fatalf("negative variance at %d: %+v", i, p)
		}
		if p.ObservedVariance < p.Variance {
			t.Fatal("observed variance must be >= latent variance")
		}
	}
}

func TestLinearKernelEndToEnd(t *testing.T) {
	x := [][]float64{{0}, {1}, {2}, {3}, {4}}
	y := []float64{0.1, 1.0, 1.9, 3.1, 3.95} // roughly y = x
	k, _ := kernel.New(kernel.Config{Type: kernel.Linear, InputDim: 1})
	opts := DefaultOptions()
	opts.NoiseVariance = 0.01
	opts.OptimizeNoise = true
	opts.MaxIterations = 30
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	prog, err := m.Train()
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(prog.FinalLML) {
		t.Fatal("NaN LML for linear kernel")
	}
	p, err := m.Predict([]float64{5})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p.Mean-5) > 0.2 {
		t.Fatalf("linear extrapolation mean = %v, want ~5", p.Mean)
	}
}
