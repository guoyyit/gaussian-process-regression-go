package kernel

import (
	"math"
	"testing"
)

func TestKernelCovarianceShapes(t *testing.T) {
	k, err := New(Config{Type: RBF, InputDim: 2, LengthScales: []float64{1.0, 2.0}})
	if err != nil {
		t.Fatal(err)
	}
	if got := k.Cov([]float64{1, 2}, []float64{1, 2}); math.Abs(got-1.0) > 1e-12 {
		t.Fatalf("diagonal cov = %v, want 1", got)
	}
	if got := k.Cov([]float64{1, 2}, []float64{10, 20}); got <= 0 || got >= 1 {
		t.Fatalf("far cov = %v, want within (0,1)", got)
	}
}

func TestKernelGradients(t *testing.T) {
	cases := []Config{
		{Type: RBF, InputDim: 2, LengthScales: []float64{1.3, 0.7}, SignalVariance: 1.6},
		{Type: Matern32, InputDim: 1, LengthScales: []float64{1.1}, SignalVariance: 2.0},
		{Type: Matern52, InputDim: 3, LengthScales: []float64{0.9}, SignalVariance: 0.5},
		{Type: Linear, InputDim: 2, BiasVariance: 0.4, ScaleVariance: 1.2},
	}
	x1 := []float64{0.8, -1.3, 2.1}
	x2 := []float64{-0.4, 0.2, 1.7}
	const eps = 1e-6
	for _, cfg := range cases {
		k, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		n := k.NumParams()
		analytic := make([]float64, n)
		k.GradCov(x1[:k.InputDim()], x2[:k.InputDim()], analytic)
		base := k.Params()
		for i := 0; i < n; i++ {
			up := append([]float64(nil), base...)
			lo := append([]float64(nil), base...)
			up[i] += eps
			lo[i] -= eps
			ku, _ := New(cfg)
			kl, _ := New(cfg)
			if err := ku.SetParams(up); err != nil {
				t.Fatal(err)
			}
			if err := kl.SetParams(lo); err != nil {
				t.Fatal(err)
			}
			numeric := (ku.Cov(x1[:k.InputDim()], x2[:k.InputDim()]) -
				kl.Cov(x1[:k.InputDim()], x2[:k.InputDim()])) / (2 * eps)
			if math.Abs(numeric-analytic[i]) > 1e-4 {
				t.Fatalf("%s param %d: analytic %v, numeric %v", cfg.Type, i, analytic[i], numeric)
			}
		}
	}
}

func TestKernelRejectsInvalidParams(t *testing.T) {
	k, _ := New(Config{Type: RBF, InputDim: 1})
	if err := k.SetParams([]float64{-1, 1}); err == nil {
		t.Fatal("expected error for negative length scale")
	}
	if _, err := New(Config{Type: Type("bogus"), InputDim: 1}); err == nil {
		t.Fatal("expected error for unknown kernel type")
	}
}
