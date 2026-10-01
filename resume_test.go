package gpr

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// errCrash 模拟训练中途进程退出。
var errCrash = errors.New("simulated crash")

func trainForResume(t *testing.T, dir string, maxIters int, crashAt int) (*OptimizeResult, error) {
	t.Helper()
	svc, err := NewService(ServiceConfig{CheckpointDir: dir, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	defer svc.Close()
	xs, ys := sinData(24, 0.0)
	cfg := defaultTrainConfig()
	cfg.Optimizer.MaxIterations = maxIters
	cfg.Optimizer.CheckpointEvery = 5
	cfg.Optimizer.NoImproveWindow = 1000
	if crashAt > 0 {
		cfg.onCheckpoint = func(iteration int, lml float64, params, mom, vel []float64) error {
			if iteration >= crashAt {
				return errCrash
			}
			return nil
		}
	}
	return svc.Train(context.Background(), xs, ys, cfg)
}

func TestCrashResumeDoesNotReapplyIters(t *testing.T) {
	dir := t.TempDir()

	// 第一次：在第 10 次迭代的检查点边界“崩溃”。
	res1, err := trainForResume(t, dir, 40, 10)
	if !errors.Is(err, errCrash) {
		t.Fatalf("want simulated crash, got res=%v err=%v", res1, err)
	}

	// 检查点尚未提交完成态；查询此时必须报错（没有完整版本）。
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cp, err := store.LoadLatest()
	if err != nil {
		t.Fatal(err)
	}
	if cp.Completed || cp.State == nil || cp.State.Iteration != 10 {
		t.Fatalf("expected incomplete checkpoint at iter 10, got completed=%v state=%+v",
			cp.Completed, cp.State)
	}
	store.Close()

	// 第二次：重启后续训到完成，不得重复已完成的 10 次迭代。
	res2, err := trainForResume(t, dir, 40, 0)
	if err != nil {
		t.Fatalf("resume training failed: %v", err)
	}
	// 总迭代 40：其中 10 次在崩溃前完成，续训只执行剩余 30 次。
	if res2.Iterations != 40 {
		t.Fatalf("total iterations after resume = %d want 40", res2.Iterations)
	}

	// 续训结果必须与一次性完成 40 次完全一致（相同状态向量续跑）。
	freshDir := t.TempDir()
	resFresh, err := trainForResume(t, freshDir, 40, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res2.FinalLML != resFresh.FinalLML {
		t.Fatalf("resumed LML %v != uninterrupted LML %v", res2.FinalLML, resFresh.FinalLML)
	}
	if res2.Kernel.Params() != resFresh.Kernel.Params() {
		t.Fatalf("resumed params %+v != uninterrupted %+v",
			res2.Kernel.Params(), resFresh.Kernel.Params())
	}
}

func TestRestoreLatestCommitted(t *testing.T) {
	dir := t.TempDir()
	if _, err := trainForResume(t, dir, 30, 0); err != nil {
		t.Fatal(err)
	}
	svc, err := NewService(ServiceConfig{CheckpointDir: dir, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	defer svc.Close()
	if err := svc.Restore(); err != nil {
		t.Fatal(err)
	}
	info, ok := svc.CurrentVersion()
	if !ok || !info.Completed || info.Iteration != 30 {
		t.Fatalf("unexpected restored version: %+v ok=%v", info, ok)
	}
	p, err := svc.Predict([]float64{1.2})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || math.IsInf(p.Mean, 0) {
		t.Fatalf("restored model gives bad prediction: %+v", p)
	}

	// 损坏的检查点必须被 Restore 显式拒绝。
	entries, _ := os.ReadDir(dir)
	for _, name := range entries {
		if filepath.Ext(name.Name()) == ckptSuffix {
			path := filepath.Join(dir, name.Name())
			raw, _ := os.ReadFile(path)
			raw[len(raw)-1] ^= 0xFF
			_ = os.WriteFile(path, raw, 0o600)
		}
	}
	svc2, _ := NewService(ServiceConfig{CheckpointDir: dir, Seed: 7})
	defer svc2.Close()
	if err := svc2.Restore(); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("want ErrCheckpointCorrupt on restore, got %v", err)
	}
}

func TestServiceNoModelErrors(t *testing.T) {
	svc, _ := NewService(ServiceConfig{})
	defer svc.Close()
	if _, err := svc.Predict([]float64{0}); err == nil {
		t.Fatal("predict before training must error")
	}
	if _, err := svc.Append(context.Background(), [][]float64{{0}}, []float64{0}); err == nil {
		t.Fatal("append before training must error")
	}
	if _, ok := svc.CurrentVersion(); ok {
		t.Fatal("no version should be visible before training")
	}
}
