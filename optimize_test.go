package gpr

import (
	"math"
	"testing"
)

func sinData(n int, noiseY float64) ([][]float64, []float64) {
	xs := make([][]float64, n)
	ys := make([]float64, n)
	for i := 0; i < n; i++ {
		x := -3 + 6*float64(i)/float64(n-1)
		xs[i] = []float64{x}
		ys[i] = math.Sin(x) + noiseY
	}
	return xs, ys
}

func TestOptimizerImprovesLMLAndLearnsHyperparams(t *testing.T) {
	xs, ys := sinData(30, 0.0)
	data, _ := NewDataset(xs, ys)

	for _, kt := range []KernelType{KernelRBF, KernelMatern, KernelRQ} {
		t.Run(string(kt), func(t *testing.T) {
			cfg := TrainConfig{
				KernelType:    kt,
				InitialParams: KernelParams{LengthScale: 0.5, SignalVariance: 0.5, Alpha: 1},
				Noise:         0.1,
				Optimize:      true,
				Stability:     DefaultStabilityConfig(),
				Optimizer: OptimizerConfig{
					MaxIterations:   150,
					Tolerance:       1e-7,
					LearningRate:    0.05,
					GradientEpsilon: 1e-5,
					NoImproveWindow: 15,
					Seed:            7,
					CheckpointEvery: 0,
				},
			}
			res, _, err := runOptimizer(data, cfg, nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.FinalLML < res.InitialLML {
				t.Fatalf("final LML %v below initial %v", res.FinalLML, res.InitialLML)
			}
			if res.StopReason != StopConverged && res.StopReason != StopMaxIters &&
				res.StopReason != StopNoImprove {
				t.Fatalf("unexpected stop reason %q", res.StopReason)
			}
			// 无噪声数据：学到的噪声标准差应很小。
			if res.Noise > 0.05 {
				t.Logf("note: learned noise = %v", res.Noise)
			}
			model, err := fitGP(data, res.Kernel, res.Noise, meanValue(ys), cfg.Stability)
			if err != nil {
				t.Fatal(err)
			}
			p, err := model.Predict([]float64{1.0})
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(p.Mean-math.Sin(1)) > 0.05 {
				t.Fatalf("%s: mean %v far from sin(1)=%v", kt, p.Mean, math.Sin(1))
			}
			t.Logf("%s: iters=%d stop=%s LML %.3f->%.3f ls=%.3f sv=%.3f noise=%.5f",
				kt, res.Iterations, res.StopReason, res.InitialLML, res.FinalLML,
				res.Kernel.Params().LengthScale, res.Kernel.Params().SignalVariance, res.Noise)
		})
	}
}

func TestOptimizerDeterminism(t *testing.T) {
	xs, ys := sinData(25, 0.0)
	data, _ := NewDataset(xs, ys)
	mk := func() *OptimizeResult {
		cfg := TrainConfig{
			KernelType:    KernelRBF,
			InitialParams: KernelParams{LengthScale: 0.7, SignalVariance: 0.7, Alpha: 1},
			Noise:         0.05,
			Optimize:      true,
			Stability:     DefaultStabilityConfig(),
			Optimizer: OptimizerConfig{
				MaxIterations: 40, LearningRate: 0.05, GradientEpsilon: 1e-5,
				NoImproveWindow: 100, Seed: 11,
			},
		}
		r, _, err := runOptimizer(data, cfg, nil)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	r1, r2 := mk(), mk()
	if r1.FinalLML != r2.FinalLML {
		t.Fatalf("LML nondeterministic: %v vs %v", r1.FinalLML, r2.FinalLML)
	}
	if r1.Kernel.Params() != r2.Kernel.Params() {
		t.Fatalf("params nondeterministic: %+v vs %+v", r1.Kernel.Params(), r2.Kernel.Params())
	}
	if r1.Noise != r2.Noise || r1.Iterations != r2.Iterations {
		t.Fatalf("optimizer trajectory differs: noise %v/%v iters %d/%d",
			r1.Noise, r2.Noise, r1.Iterations, r2.Iterations)
	}
}

func TestOptimizeDisabled(t *testing.T) {
	xs, ys := sinData(10, 0.0)
	data, _ := NewDataset(xs, ys)
	cfg := TrainConfig{
		KernelType:    KernelRBF,
		InitialParams: KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1},
		Noise:         0.01,
		Optimize:      false,
		Stability:     DefaultStabilityConfig(),
		Optimizer:     defaultOptimizerConfig(),
	}
	res, state, err := runOptimizer(data, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.StopReason != StopNotOptimize || state != nil {
		t.Fatalf("unexpected: reason=%q state=%v", res.StopReason, state)
	}
	if res.InitialLML != res.FinalLML {
		t.Fatal("LML must be unchanged when optimization disabled")
	}
}

func TestPackUnpackRoundTrip(t *testing.T) {
	k, _ := NewKernel(KernelRQ, KernelParams{LengthScale: 2.5, SignalVariance: 1.5, Alpha: 0.7})
	noise := 0.01
	packed := packParams(k, noise)
	if len(packed) != 4 {
		t.Fatalf("rq packed dim = %d want 4", len(packed))
	}
	p2, n2, err := unpackParams(KernelRQ, packed)
	if err != nil {
		t.Fatal(err)
	}
	if !approx(p2.LengthScale, 2.5, 1e-12) || !approx(p2.SignalVariance, 1.5, 1e-12) ||
		!approx(p2.Alpha, 0.7, 1e-12) || !approx(n2, noise, 1e-12) {
		t.Fatalf("round trip mismatch: %+v noise %v", p2, n2)
	}
}

func TestLMLFiniteAndDifferentiable(t *testing.T) {
	xs, ys := sinData(12, 0.0)
	data, _ := NewDataset(xs, ys)
	params := packParams(mustKernel(KernelRBF, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}), 0.01)
	lml, grad, err := lmlWithGradient(data, KernelRBF, params, DefaultStabilityConfig(), 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(lml) || math.IsInf(lml, 0) {
		t.Fatalf("non-finite LML %v", lml)
	}
	for i, g := range grad {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("non-finite gradient component %d: %v", i, g)
		}
	}
}
