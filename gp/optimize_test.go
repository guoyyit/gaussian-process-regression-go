package gp

import (
	"math"
	"testing"

	"github.com/guoyyit/gaussian-process-regression-go/kernel"
)

func sincData(n int) ([][]float64, []float64) {
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		v := -5 + 10*float64(i)/float64(n-1)
		x[i] = []float64{v}
		target := 1.0
		if math.Abs(v) > 1e-12 {
			target = math.Sin(v) / v
		}
		y[i] = target
	}
	return x, y
}

func TestLMLGradientFiniteDifference(t *testing.T) {
	x, y := sincData(8)
	for _, kt := range []kernel.Type{kernel.RBF, kernel.Matern32, kernel.Matern52, kernel.Linear} {
		k, _ := kernel.New(kernel.Config{Type: kt, InputDim: 1, LengthScales: []float64{1.3}, SignalVariance: 1.4, BiasVariance: 0.7, ScaleVariance: 0.9})
		opts := DefaultOptions()
		opts.NoiseVariance = 0.01
		opts.OptimizeNoise = true
		m, err := NewModel(k, x, y, opts)
		if err != nil {
			t.Fatal(err)
		}
		z := encodeLogParams(k.Params(), opts.NoiseVariance, true)
		f, gz, _, err := m.negLML(append([]float64(nil), z...), true)
		if err != nil {
			t.Fatal(err)
		}
		const eps = 1e-6
		for i := range z {
			up := append([]float64(nil), z...)
			lo := append([]float64(nil), z...)
			up[i] += eps
			lo[i] -= eps
			fu, _, _, _ := m.negLML(up, true)
			fl, _, _, _ := m.negLML(lo, true)
			numeric := (fu - fl) / (2 * eps)
			if math.Abs(numeric-gz[i]) > 1e-3*(1+math.Abs(gz[i])) {
				t.Fatalf("%s log-param %d: analytic %v numeric %v (f=%v)", kt, i, gz[i], numeric, f)
			}
		}
	}
}

func TestOptimizationImprovesLML(t *testing.T) {
	x, y := sincData(25)
	for _, kt := range []kernel.Type{kernel.RBF, kernel.Matern32, kernel.Matern52} {
		k, _ := kernel.New(kernel.Config{Type: kt, InputDim: 1, LengthScales: []float64{3.0}, SignalVariance: 2.0})
		opts := DefaultOptions()
		opts.NoiseVariance = 0.1
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
		if math.IsNaN(prog.FinalLML) {
			t.Fatalf("%s: NaN final LML", kt)
		}
		if prog.FinalLML < prog.InitialLML-1e-9 {
			t.Fatalf("%s: final LML %v below initial %v", kt, prog.FinalLML, prog.InitialLML)
		}
		if prog.FinalLML <= prog.InitialLML+1e-6 {
			t.Fatalf("%s: expected LML improvement, initial %v final %v", kt, prog.InitialLML, prog.FinalLML)
		}
		if prog.Termination == "" || prog.Iterations == 0 {
			t.Fatalf("%s: missing termination info", kt)
		}
		if len(prog.CompletedSteps) != prog.Iterations {
			t.Fatalf("%s: recorded %d steps for %d iterations", kt, len(prog.CompletedSteps), prog.Iterations)
		}
	}
}

func TestOptimizationNeverBelowInitialWhenNoImprovement(t *testing.T) {
	// Data with extreme initial length scale; result must not regress.
	x, y := sincData(12)
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{0.5}, SignalVariance: 1.0})
	opts := DefaultOptions()
	opts.NoiseVariance = 0.05
	opts.MaxIterations = 5
	m, _ := NewModel(k, x, y, opts)
	prog, err := m.Train()
	if err != nil {
		t.Fatal(err)
	}
	if prog.FinalLML < prog.InitialLML-1e-9 {
		t.Fatalf("final LML %v below initial %v", prog.FinalLML, prog.InitialLML)
	}
}

func TestOptimizationMaxIterationsCap(t *testing.T) {
	x, y := sincData(20)
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{2.5}})
	opts := DefaultOptions()
	opts.NoiseVariance = 0.05
	opts.OptimizeNoise = true
	opts.MaxIterations = 3
	opts.ConvergenceTol = 0
	opts.GradientTol = 0
	m, _ := NewModel(k, x, y, opts)
	prog, err := m.Train()
	if err != nil {
		t.Fatal(err)
	}
	if prog.Iterations != 3 {
		t.Fatalf("iterations = %d, want 3", prog.Iterations)
	}
	if prog.Termination != TerminatedMaxIters {
		t.Fatalf("termination = %q, want max_iterations", prog.Termination)
	}
}

func TestResumeDoesNotReapplyIterations(t *testing.T) {
	x, y := sincData(20)
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{2.5}})
	opts := DefaultOptions()
	opts.NoiseVariance = 0.05
	opts.OptimizeNoise = true
	opts.MaxIterations = 3
	opts.ConvergenceTol = 0
	opts.GradientTol = 0
	m, _ := NewModel(k, x, y, opts)
	first, err := m.Train()
	if err != nil {
		t.Fatal(err)
	}

	k2, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{2.5}})
	m2, _ := NewModel(k2, x, y, opts)
	second, err := m2.Resume(*first)
	if err != nil {
		t.Fatal(err)
	}
	// Resuming runs at most 3 NEW iterations and starts at iteration 3.
	if second.Iterations < 3 || second.Iterations > 6 {
		t.Fatalf("resume iteration count = %d, want 3..6", second.Iterations)
	}
	if len(second.CompletedSteps) != second.Iterations {
		t.Fatalf("steps %d != iterations %d", len(second.CompletedSteps), second.Iterations)
	}
	if second.CompletedSteps[0].Iteration != 1 {
		t.Fatal("checkpointed steps were not preserved on resume")
	}
}

func TestTrainedModelPredictsWell(t *testing.T) {
	x, y := sincData(30)
	k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{2.0}})
	opts := DefaultOptions()
	opts.NoiseVariance = 0.01
	opts.OptimizeNoise = true
	opts.MaxIterations = 80
	m, _ := NewModel(k, x, y, opts)
	if _, err := m.Train(); err != nil {
		t.Fatal(err)
	}
	// sinc(2) ~= 0.45465
	p, err := m.Predict([]float64{2})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p.Mean-0.45465) > 0.05 {
		t.Fatalf("trained mean at 2 = %v, want ~0.4547", p.Mean)
	}
}
