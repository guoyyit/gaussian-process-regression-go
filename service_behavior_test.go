package gpr

import (
	"context"
	"math"
	"sync"
	"testing"
)

func trainService(t *testing.T, svc *Service, xs [][]float64, ys []float64, iters int) *OptimizeResult {
	t.Helper()
	cfg := defaultTrainConfig()
	cfg.Optimizer.MaxIterations = iters
	cfg.Optimizer.CheckpointEvery = 10
	res, err := svc.Train(context.Background(), xs, ys, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestConcurrentReadsDuringUpdate(t *testing.T) {
	dir := t.TempDir()
	svc, _ := NewService(ServiceConfig{CheckpointDir: dir, Seed: 3})
	defer svc.Close()

	xs1, ys1 := sinData(20, 0.0)
	trainService(t, svc, xs1, ys1, 30)

	// 追加数据，制造一次较长的更新。
	extra := make([][]float64, 15)
	extraY := make([]float64, 15)
	for i := range extra {
		x := 3 + 3*float64(i)/14
		extra[i] = []float64{x}
		extraY[i] = math.Sin(x)
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// 多个读者在更新期间持续查询：任何结果都必须来自某个完整版本且有限。
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					p, err := svc.Predict([]float64{0.5})
					if err != nil {
						t.Errorf("concurrent predict error: %v", err)
						return
					}
					if math.IsNaN(p.Mean) || math.IsNaN(p.Variance) ||
						math.IsInf(p.Mean, 0) || p.Variance < 0 {
						t.Errorf("torn read: %+v", p)
						return
					}
				}
			}
		}()
	}

	res, err := svc.Append(context.Background(), extra, extraY)
	if err != nil {
		t.Fatal(err)
	}
	close(stop)
	wg.Wait()

	if res.Iterations == 0 {
		t.Fatal("append did not retrain")
	}
	info, ok := svc.CurrentVersion()
	if !ok || !info.Completed {
		t.Fatalf("post-append version not committed: %+v", info)
	}
}

func TestAppendMatchesFullRetrain(t *testing.T) {
	xs, ys := sinData(20, 0.0)
	extra := [][]float64{{2.5}, {2.8}, {3.0}}
	extraY := []float64{math.Sin(2.5), math.Sin(2.8), math.Sin(3.0)}

	// A：先训练 20 点，再追加 3 点。
	svcA, _ := NewService(ServiceConfig{CheckpointDir: t.TempDir(), Seed: 99})
	defer svcA.Close()
	trainService(t, svcA, xs, ys, 60)
	if _, err := svcA.Append(context.Background(), extra, extraY); err != nil {
		t.Fatal(err)
	}

	// B：直接用全部 23 点训练。
	full := make([][]float64, 0, 23)
	fullY := make([]float64, 0, 23)
	full = append(full, xs...)
	fullY = append(fullY, ys...)
	full = append(full, extra...)
	fullY = append(fullY, extraY...)
	svcB, _ := NewService(ServiceConfig{CheckpointDir: t.TempDir(), Seed: 99})
	defer svcB.Close()
	trainService(t, svcB, full, fullY, 60)

	// 追加结果与全量重训逐位一致（同样的数据、初始参数、种子与迭代数）。
	infoA, _ := svcA.CurrentVersion()
	infoB, _ := svcB.CurrentVersion()
	if infoA.FinalLML != infoB.FinalLML {
		t.Fatalf("LML mismatch: append=%v full=%v", infoA.FinalLML, infoB.FinalLML)
	}
	query := []float64{-2.2, 0.3, 1.7, 2.6, 4.0}
	queryPts := make([][]float64, len(query))
	for i, v := range query {
		queryPts[i] = []float64{v}
	}
	pa, err := svcA.PredictBatch(queryPts)
	if err != nil {
		t.Fatal(err)
	}
	pb, err := svcB.PredictBatch(queryPts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range query {
		if pa[i].Mean != pb[i].Mean || pa[i].Variance != pb[i].Variance {
			t.Fatalf("query %v mismatch: append=%+v full=%+v",
				query[i], pa[i], pb[i])
		}
	}

	// 增量后观测点均值应逼近观测值。
	p, err := svcA.Predict([]float64{2.8})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p.Mean-math.Sin(2.8)) > 1e-4 {
		t.Fatalf("appended observed mean %v far from target %v", p.Mean, math.Sin(2.8))
	}
}

func TestAppendDimensionMismatch(t *testing.T) {
	svc, _ := NewService(ServiceConfig{})
	defer svc.Close()
	xs, ys := sinData(8, 0.0)
	trainService(t, svc, xs, ys, 5)
	if _, err := svc.Append(context.Background(), [][]float64{{1, 2}}, []float64{1}); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

func TestDeterministicAcrossProcesses(t *testing.T) {
	// 不依赖任何全局随机状态：两个独立服务、相同种子得到逐位相同预测。
	xs, ys := sinData(18, 0.0)
	run := func(dir string) Prediction {
		svc, _ := NewService(ServiceConfig{CheckpointDir: dir, Seed: 123})
		defer svc.Close()
		trainService(t, svc, xs, ys, 45)
		p, err := svc.Predict([]float64{-0.7})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	p1 := run(t.TempDir())
	p2 := run(t.TempDir())
	if p1 != p2 {
		t.Fatalf("nondeterministic across fresh services: %+v vs %+v", p1, p2)
	}
}

func TestSequentialAppendsBuildVersions(t *testing.T) {
	svc, _ := NewService(ServiceConfig{CheckpointDir: t.TempDir(), Seed: 5})
	defer svc.Close()

	xs := [][]float64{{0}, {1}, {2}}
	ys := []float64{0, 1, 2}
	trainService(t, svc, xs, ys, 20)

	var wg sync.WaitGroup
	// 并发触发多次追加：每次都基于已提交版本构建新版本，最终版本必须完整可用。
	for i := 0; i < 4; i++ {
		wg.Add(1)
		i := i
		go func() {
			defer wg.Done()
			x := 3 + float64(i)
			_, err := svc.Append(context.Background(), [][]float64{{x}}, []float64{x})
			if err != nil {
				t.Errorf("append %d: %v", i, err)
			}
		}()
	}
	wg.Wait()

	info, ok := svc.CurrentVersion()
	if !ok || !info.Completed {
		t.Fatalf("final version not healthy: %+v ok=%v", info, ok)
	}
	p, err := svc.Predict([]float64{4.5})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || p.Variance < 0 {
		t.Fatalf("bad prediction after concurrent appends: %+v", p)
	}

	// 并发追加不保证合并次数（后来的训练可能基于较早快照），但每次提交都必须自洽。
	// 串行再追加一次，确保模型仍能正常增长。
	if _, err := svc.Append(context.Background(), [][]float64{{7}}, []float64{7}); err != nil {
		t.Fatal(err)
	}
}

func TestDefaultSeedSet(t *testing.T) {
	svc, err := NewService(ServiceConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer svc.Close()
	if svc.cfg.Seed != 1 {
		t.Fatalf("default seed = %d want 1", svc.cfg.Seed)
	}
}
