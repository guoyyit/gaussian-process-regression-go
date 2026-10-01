package gp

import (
	"math"
	"testing"

	"github.com/guoyyit/gaussian-process-regression-go/kernel"
)

func train1D(t *testing.T, noise float64, stability StabilityStrategy) *Model {
	t.Helper()
	k, err := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{1.0}})
	if err != nil {
		t.Fatal(err)
	}
	x := [][]float64{{0}, {1}, {2}, {3}, {4}}
	y := []float64{0, 0.8415, 0.9093, 0.1411, -0.7568}
	opts := DefaultOptions()
	opts.NoiseVariance = noise
	opts.Stability = stability
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestPredictVarianceProperties(t *testing.T) {
	noise := 0.05
	m := train1D(t, noise, StabilityJitter)
	// Observation points: variance close to noise level.
	for i, xi := range m.X {
		p, err := m.Predict(xi)
		if err != nil {
			t.Fatal(err)
		}
		if p.Variance < 0 {
			t.Fatalf("negative variance %v", p.Variance)
		}
		if math.Abs(p.Variance-noise) > 1e-2 {
			t.Fatalf("obs point %d variance %v, want ~%v", i, p.Variance, noise)
		}
	}
	// Far from data: variance grows toward prior variance + noise.
	pfar, err := m.Predict([]float64{100})
	if err != nil {
		t.Fatal(err)
	}
	pnear, _ := m.Predict([]float64{0.5})
	if !(pfar.Variance > pnear.Variance) {
		t.Fatalf("far variance %v should exceed near variance %v", pfar.Variance, pnear.Variance)
	}
	if pfar.Variance > 1+1e-9 {
		t.Fatalf("far variance %v exceeds prior variance 1", pfar.Variance)
	}
}

func TestTinyNoiseInterpolatesObservations(t *testing.T) {
	m := train1D(t, 1e-10, StabilityJitter)
	for i, xi := range m.X {
		p, err := m.Predict(xi)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(p.Mean-m.Y[i]) > 1e-4 {
			t.Fatalf("mean %v at obs %v, want %v", p.Mean, xi, m.Y[i])
		}
	}
}

func TestMultiDimensionalInput(t *testing.T) {
	k, _ := kernel.New(kernel.Config{Type: kernel.Matern32, InputDim: 2})
	x := [][]float64{{0, 0}, {1, 1}, {2, 0}}
	y := []float64{1, 2, 1.5}
	opts := DefaultOptions()
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict([]float64{0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || math.IsNaN(p.Variance) {
		t.Fatal("NaN prediction")
	}
	if p.Variance < 0 {
		t.Fatal("negative variance")
	}
}

func TestDuplicatePointsStrictModeErrors(t *testing.T) {
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{1.0}})
	x := [][]float64{{1}, {1}}
	y := []float64{0.5, 0.5}
	opts := DefaultOptions()
	opts.NoiseVariance = 1e-300
	opts.Stability = StabilityStrict
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Predict([]float64{1}); err == nil {
		t.Fatal("expected identifiable error for singular matrix in strict mode")
	}
}

func TestDuplicatePointsJitterRecovers(t *testing.T) {
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{1.0}})
	x := [][]float64{{1}, {1}}
	y := []float64{0.5, 0.5}
	opts := DefaultOptions()
	opts.NoiseVariance = 1e-300
	opts.Stability = StabilityJitter
	m, err := NewModel(k, x, y, opts)
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict([]float64{1})
	if err != nil {
		t.Fatalf("jitter should recover: %v", err)
	}
	if math.IsNaN(p.Mean) || math.IsNaN(p.Variance) {
		t.Fatal("NaN despite jitter")
	}
	if m.AppliedJitter() == 0 {
		t.Fatal("expected positive applied jitter")
	}
}

func TestInvalidInputRejected(t *testing.T) {
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1})
	if _, err := NewModel(k, nil, nil, DefaultOptions()); err == nil {
		t.Fatal("expected empty-data error")
	}
	if _, err := NewModel(k, [][]float64{{1}}, []float64{1, 2}, DefaultOptions()); err == nil {
		t.Fatal("expected length mismatch error")
	}
}
