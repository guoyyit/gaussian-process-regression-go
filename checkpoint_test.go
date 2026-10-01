package gpr

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func sampleCheckpoint(iter int, completed bool) *Checkpoint {
	k, _ := NewKernel(KernelRBF, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1})
	return &Checkpoint{
		Version:    checkpointVersion,
		ModelID:    "model-xyz",
		KernelType: KernelRBF,
		Dim:        1,
		X:          [][]float64{{1}, {2}},
		Y:          []float64{3, 4},
		Noise:      0.01,
		MeanTrain:  3.5,
		Result: OptimizeResult{
			Kernel: k, Noise: 0.01, InitialLML: 1, FinalLML: 2,
			Iterations: iter, StopReason: StopMaxIters,
			History: []IterationStat{{Iteration: iter, LML: 2,
				Params: KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}, Noise: 0.01}},
		},
		State:     &optimizerState{Iteration: iter, LML: 2, Params: []float64{0, 0, -4.6}},
		Completed: completed,
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	cp := sampleCheckpoint(12, false)
	raw, err := encodeCheckpoint(cp)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeCheckpoint(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelID != cp.ModelID || got.Result.Iterations != 12 || got.Completed {
		t.Fatalf("round trip mismatch: %+v", got)
	}
	if got.Result.Kernel.Type() != KernelRBF {
		t.Fatal("kernel type lost through encoding")
	}
	if got.State == nil || got.State.Iteration != 12 || len(got.State.Params) != 3 {
		t.Fatalf("optimizer state lost: %+v", got.State)
	}
}

func TestCheckpointCorruptionRejected(t *testing.T) {
	cp := sampleCheckpoint(1, true)
	raw, _ := encodeCheckpoint(cp)

	// 翻转载荷区最后一个字节，SHA-256 校验必须失败。
	corrupt := append([]byte(nil), raw...)
	corrupt[len(corrupt)-1] ^= 0xFF
	if _, err := decodeCheckpoint(corrupt); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("flipped payload: want ErrCheckpointCorrupt, got %v", err)
	}

	// 截断。
	if _, err := decodeCheckpoint(raw[:len(raw)/2]); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("truncated: want ErrCheckpointCorrupt, got %v", err)
	}

	// 错误魔数。
	badMagic := append([]byte(nil), raw...)
	badMagic[0] = 'X'
	if _, err := decodeCheckpoint(badMagic); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("bad magic: want ErrCheckpointCorrupt, got %v", err)
	}

	// 随机垃圾。
	if _, err := decodeCheckpoint([]byte("not a checkpoint at all")); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("garbage: want ErrCheckpointCorrupt, got %v", err)
	}
}

func TestFileStoreSaveLoad(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Save(sampleCheckpoint(5, false)); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(sampleCheckpoint(10, false)); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(sampleCheckpoint(10, true)); err != nil {
		t.Fatal(err)
	}
	got, err := store.LoadLatest()
	if err != nil {
		t.Fatal(err)
	}
	if !got.Completed || got.Result.Iterations != 10 {
		t.Fatalf("expected latest completed checkpoint, got %+v", got)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	// 不应残留临时文件。
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if filepath.Ext(e.Name()) == tempSuffix {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
}

func TestFileStoreCorruptFileRejected(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileStore(dir)
	if err := store.Save(sampleCheckpoint(3, false)); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(dir)
	var ckpt string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ckptSuffix {
			ckpt = e.Name()
		}
	}
	if ckpt == "" {
		t.Fatal("no checkpoint file written")
	}
	p := filepath.Join(dir, ckpt)
	raw, _ := os.ReadFile(p)
	raw[20] ^= 0xFF
	if err := os.WriteFile(p, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadLatest(); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("want ErrCheckpointCorrupt, got %v", err)
	}
}

func TestEmptyStore(t *testing.T) {
	store, _ := NewFileStore(t.TempDir())
	if _, err := store.LoadLatest(); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("want ErrNoCheckpoint, got %v", err)
	}
}
