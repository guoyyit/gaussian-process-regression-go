package service

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/guoyyit/gaussian-process-regression-go/gp"
	"github.com/guoyyit/gaussian-process-regression-go/kernel"
	"github.com/guoyyit/gaussian-process-regression-go/store"
)

func baseOpts() gp.Options {
	o := gp.DefaultOptions()
	o.NoiseVariance = 0.02
	o.OptimizeNoise = true
	o.MaxIterations = 30
	o.Stability = gp.StabilityJitter
	o.RandomSeed = 7
	return o
}

func rbfCfg(ls float64) kernel.Config {
	return kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{ls}}
}

func sincTrainData() ([][]float64, []float64) {
	x := make([][]float64, 12)
	y := make([]float64, 12)
	for i := 0; i < 12; i++ {
		v := -3 + 6*float64(i)/11
		x[i] = []float64{v}
		target := 1.0
		if math.Abs(v) > 1e-12 {
			target = math.Sin(v) / v
		}
		y[i] = target
	}
	return x, y
}

func newTestService(t *testing.T) (*Service, string) {
	t.Helper()
	dir := t.TempDir()
	st, err := store.NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	return New(st), dir
}

func trainOne(t *testing.T, svc *Service, id string, x [][]float64, y []float64) *VersionedModel {
	t.Helper()
	v, err := svc.Train(context.Background(), TrainRequest{
		ModelID: id,
		Kernel:  rbfCfg(1.5),
		X:       x,
		Y:       y,
		Options: baseOpts(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestTrainPredictVersioned(t *testing.T) {
	svc, _ := newTestService(t)
	x, y := sincTrainData()
	v := trainOne(t, svc, "m", x, y)
	if v.Version != 1 {
		t.Fatalf("version = %d, want 1", v.Version)
	}
	p, err := svc.Predict("m", []float64{0.5})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || p.Variance < 0 {
		t.Fatalf("bad prediction %+v", p)
	}
}

func TestDeterminismAcrossRuns(t *testing.T) {
	x, y := sincTrainData()
	var first, second *gp.Model
	{
		svc, _ := newTestService(t)
		v := trainOne(t, svc, "m", x, y)
		first = v.Model
	}
	{
		svc, _ := newTestService(t)
		v := trainOne(t, svc, "m", x, y)
		second = v.Model
	}
	p1 := first.Kernel.Params()
	p2 := second.Kernel.Params()
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("kernel param %d differs: %v vs %v", i, p1[i], p2[i])
		}
	}
	if first.Noise != second.Noise {
		t.Fatalf("noise differs: %v vs %v", first.Noise, second.Noise)
	}
	pr1, _ := first.Predict([]float64{1.23})
	pr2, _ := second.Predict([]float64{1.23})
	if pr1.Mean != pr2.Mean || pr1.Variance != pr2.Variance {
		t.Fatalf("predictions differ: %+v vs %+v", pr1, pr2)
	}
}

func TestConcurrentReadsDuringUpdate(t *testing.T) {
	svc, _ := newTestService(t)
	x, y := sincTrainData()
	trainOne(t, svc, "m", x, y)

	// Append in one goroutine; hammer reads concurrently. Reads must only
	// ever see complete versions (v1 during append, v2 afterward).
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := svc.AppendObservations(context.Background(), "m",
			[][]float64{{3.5}, {4.0}}, []float64{math.Sin(3.5) / 3.5, math.Sin(4.0) / 4.0},
			baseOpts())
		if err != nil {
			t.Errorf("append: %v", err)
		}
		close(stop)
	}()

	for i := 0; i < 200; i++ {
		v, err := svc.Current("m")
		if err != nil {
			t.Fatal(err)
		}
		if v.Version < 1 || v.Version > 2 {
			t.Fatalf("observed invalid version %d", v.Version)
		}
		p, err := v.Model.Predict([]float64{float64(i%5 - 2)})
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(p.Mean) || math.IsNaN(p.Variance) {
			t.Fatal("NaN under concurrent access")
		}
	}
	<-stop
	wg.Wait()

	v, _ := svc.Current("m")
	if v.Version != 2 {
		t.Fatalf("version after append = %d, want 2", v.Version)
	}
}

func TestAppendMatchesFullRetrain(t *testing.T) {
	x, y := sincTrainData()
	// Scenario A: train on all 14 points at once.
	svcA, _ := newTestService(t)
	fullX := append(clone(x), []float64{3.5}, []float64{4.0})
	fullY := append(append([]float64(nil), y...), math.Sin(3.5)/3.5, math.Sin(4.0)/4.0)
	va := trainOne(t, svcA, "full", fullX, fullY)

	// Scenario B: train on 12 then append 2.
	svcB, _ := newTestService(t)
	trainOne(t, svcB, "inc", x, y)
	vb, err := svcB.AppendObservations(context.Background(), "inc",
		[][]float64{{3.5}, {4.0}}, []float64{math.Sin(3.5) / 3.5, math.Sin(4.0) / 4.0},
		baseOpts())
	if err != nil {
		t.Fatal(err)
	}

	pa := va.Model.Kernel.Params()
	pb := vb.Model.Kernel.Params()
	for i := range pa {
		if math.Abs(pa[i]-pb[i]) > 1e-6*math.Max(1, math.Abs(pa[i])) {
			t.Fatalf("kernel param %d: full %v vs append %v", i, pa[i], pb[i])
		}
	}
	for _, q := range []float64{-2.5, 0, 1.7, 3.8} {
		pa2, _ := va.Model.Predict([]float64{q})
		pb2, _ := vb.Model.Predict([]float64{q})
		if math.Abs(pa2.Mean-pb2.Mean) > 1e-6 {
			t.Fatalf("mean at %v: full %v vs append %v", q, pa2.Mean, pb2.Mean)
		}
		if math.Abs(pa2.Variance-pb2.Variance) > 1e-6 {
			t.Fatalf("var at %v: full %v vs append %v", q, pa2.Variance, pb2.Variance)
		}
	}
}

func TestCrashResumeContinuesOnce(t *testing.T) {
	dir := t.TempDir()
	st, _ := store.NewFileStore(dir)
	svc := New(st)
	x, y := sincTrainData()

	opts := baseOpts()
	opts.MaxIterations = 2
	opts.ConvergenceTol = 0
	opts.GradientTol = 0

	// Train 2 iterations with per-iteration checkpointing.
	v, err := svc.Train(context.Background(), TrainRequest{
		ModelID: "m", Kernel: rbfCfg(1.5), X: x, Y: y,
		Options: opts, CheckpointEvery: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v.Model.Progress().Iterations != 2 {
		t.Fatalf("expected 2 iterations, got %d", v.Model.Progress().Iterations)
	}

	// New service instance simulates a process restart.
	svc2 := New(st)
	restored, err := svc2.Restore(context.Background(), "m", rbfCfg(1.5), true)
	if err != nil {
		t.Fatal(err)
	}
	prog := restored.Model.Progress()
	// Only NEW iterations were run; completed steps 1..2 were preserved.
	if prog.Iterations < 2 {
		t.Fatalf("resume iterations = %d, expected at least 2 total", prog.Iterations)
	}
	if len(prog.CompletedSteps) < 2 || prog.CompletedSteps[0].Iteration != 1 {
		t.Fatalf("completed iterations were re-applied: %+v", prog.CompletedSteps)
	}
	for i := 1; i < len(prog.CompletedSteps); i++ {
		if prog.CompletedSteps[i].Iteration <= prog.CompletedSteps[i-1].Iteration {
			t.Fatal("iteration numbers must strictly increase")
		}
	}
}

func TestCorruptCheckpointRefusedOnRestore(t *testing.T) {
	svc, _ := newTestService(t)
	x, y := sincTrainData()
	trainOne(t, svc, "m", x, y)

	// Create a second service on a deliberately corrupt checkpoint dir.
	badDir := t.TempDir()
	if err := writeFile(filepath.Join(badDir, "checkpoint.json"), "{not json"); err != nil {
		t.Fatal(err)
	}
	badStore, _ := store.NewFileStore(badDir)
	svcBad := New(badStore)
	if _, err := svcBad.Restore(context.Background(), "m", rbfCfg(1.5), false); !errors.Is(err, store.ErrCheckpointCorrupt) {
		t.Fatalf("restore err = %v, want ErrCheckpointCorrupt", err)
	}
}

func TestModelNotFound(t *testing.T) {
	svc, _ := newTestService(t)
	if _, err := svc.Predict("ghost", []float64{0}); err == nil {
		t.Fatal("expected not-found error")
	}
}

func TestCrashResumeTwiceIsMonotoneAndNonRepeating(t *testing.T) {
	dir := t.TempDir()
	st, _ := store.NewFileStore(dir)
	x, y := sincTrainData()

	opts := baseOpts()
	opts.MaxIterations = 1
	opts.ConvergenceTol = 0
	opts.GradientTol = 0

	runOnce := func() *gp.OptimizeProgress {
		svc := New(st)
		v, err := svc.Restore(context.Background(), "m", rbfCfg(1.5), true)
		if err != nil {
			// First run: no checkpoint yet, train from scratch.
			v, err = svc.Train(context.Background(), TrainRequest{
				ModelID: "m", Kernel: rbfCfg(1.5), X: x, Y: y,
				Options: opts, CheckpointEvery: 1,
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		return v.Model.Progress()
	}

	p1 := runOnce()
	p2 := runOnce()
	p3 := runOnce()
	if p1.Iterations != 1 || p2.Iterations != 2 || p3.Iterations != 3 {
		t.Fatalf("iterations across restarts = %d,%d,%d, want 1,2,3", p1.Iterations, p2.Iterations, p3.Iterations)
	}
	if p2.FinalLML < p1.InitialLML-1e-9 || p3.FinalLML < p2.FinalLML-1e-9 {
		t.Fatal("LML regressed across crash restarts")
	}
	// No iteration number is repeated in the final record.
	seen := map[int]bool{}
	for _, s := range p3.CompletedSteps {
		if seen[s.Iteration] {
			t.Fatalf("iteration %d applied more than once", s.Iteration)
		}
		seen[s.Iteration] = true
	}
}

func clone(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = append([]float64(nil), x[i]...)
	}
	return out
}

func writeFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}
