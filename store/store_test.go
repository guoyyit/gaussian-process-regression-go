package store

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func sampleData() *ModelData {
	return &ModelData{
		SchemaVersion: SchemaVersion,
		ModelID:       "m1",
		KernelType:    "rbf",
		InputDim:      1,
		KernelParams:  []float64{1.2, 0.8},
		Noise:         0.01,
		X:             [][]float64{{0}, {1}, {2}},
		Y:             []float64{0, 1, 0.5},
		Options:       OptionsData{Stability: "jitter"},
		Progress: ProgressData{
			InitialLML:    1.25,
			FinalLML:      2.5,
			Iterations:    4,
			Converged:     true,
			Termination:   "converged",
			FinalParams:   []float64{1.2, 0.8, 0.01},
			InitialParams: []float64{1, 1, 0.1},
			CompletedSteps: []StepData{
				{Iteration: 1, Params: []float64{1.1, 0.9, 0.05}, NegLML: -1.5, GradNorm: 0.2},
			},
		},
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	st, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := st.Load(ctx); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("empty store err = %v, want ErrNoCheckpoint", err)
	}
	data := sampleData()
	if err := st.Save(ctx, data); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelID != "m1" || got.Progress.Iterations != 4 || got.Progress.FinalLML != 2.5 {
		t.Fatalf("roundtrip mismatch: %+v", got)
	}
	if len(got.Progress.CompletedSteps) != 1 || got.Progress.CompletedSteps[0].Iteration != 1 {
		t.Fatal("steps did not roundtrip")
	}
}

func TestCorruptCheckpointRejected(t *testing.T) {
	dir := t.TempDir()
	st, _ := NewFileStore(dir)
	ctx := context.Background()
	if err := st.Save(ctx, sampleData()); err != nil {
		t.Fatal(err)
	}

	cp := filepath.Join(dir, checkpointName)

	// 1) Truncated (simulates half-written content despite rename).
	raw, _ := os.ReadFile(cp)
	if err := os.WriteFile(cp, raw[:len(raw)/2], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Load(ctx); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("truncated: err = %v, want ErrCheckpointCorrupt", err)
	}

	// 2) Tampered payload, intact JSON.
	if err := st.Save(ctx, sampleData()); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(cp)
	tampered := append([]byte(nil), raw...)
	tampered[len(tampered)-40] ^= 0xFF
	if err := os.WriteFile(cp, tampered, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Load(ctx); !errors.Is(err, ErrCheckpointCorrupt) {
		t.Fatalf("tampered: err = %v, want ErrCheckpointCorrupt", err)
	}

	// 3) Semantically invalid payload.
	bad := sampleData()
	bad.InputDim = 2
	if err := st.Save(ctx, bad); err == nil {
		t.Fatal("expected Save to reject dimension mismatch")
	}
}

func TestCrashLeavesTempButLoadsPrevious(t *testing.T) {
	dir := t.TempDir()
	st, _ := NewFileStore(dir)
	ctx := context.Background()
	if err := st.Save(ctx, sampleData()); err != nil {
		t.Fatal(err)
	}
	// Simulate a crash mid-write: a leftover temp file with junk.
	if err := os.WriteFile(filepath.Join(dir, tempName), []byte("{not finished"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load(ctx)
	if err != nil {
		t.Fatalf("leftover temp must not break loading the committed checkpoint: %v", err)
	}
	if got.ModelID != "m1" {
		t.Fatal("loaded wrong model")
	}
}

func TestClear(t *testing.T) {
	dir := t.TempDir()
	st, _ := NewFileStore(dir)
	ctx := context.Background()
	if err := st.Save(ctx, sampleData()); err != nil {
		t.Fatal(err)
	}
	if err := st.Clear(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Load(ctx); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("after clear err = %v", err)
	}
}
