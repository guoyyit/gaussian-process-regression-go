package store

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/guoyyit/gaussian-process-regression-go/gp"
)

const (
	// SchemaVersion is bumped when the on-disk format changes.
	SchemaVersion  = 1
	checkpointName = "checkpoint.json"
	tempName       = "checkpoint.json.tmp"
	checksumAlgo   = "sha256"
)

// ErrCheckpointCorrupt is returned when persisted data fails integrity
// validation; such checkpoints are never loaded.
var ErrCheckpointCorrupt = errors.New("checkpoint is corrupt")

// ErrNoCheckpoint is returned when no complete checkpoint exists.
var ErrNoCheckpoint = errors.New("no complete checkpoint")

// envelope wraps the payload with an integrity tag so that truncated or
// half-written files can be detected on load.
type envelope struct {
	ChecksumAlgo string          `json:"checksum_algo"`
	Checksum     string          `json:"checksum"`
	Payload      json.RawMessage `json:"payload"`
}

// ModelData is the serializable representation of a trained model.
type ModelData struct {
	SchemaVersion int          `json:"schema_version"`
	ModelID       string       `json:"model_id"`
	KernelType    string       `json:"kernel_type"`
	InputDim      int          `json:"input_dim"`
	KernelParams  []float64    `json:"kernel_params"`
	Noise         float64      `json:"noise"`
	X             [][]float64  `json:"x"`
	Y             []float64    `json:"y"`
	Options       OptionsData  `json:"options"`
	Progress      ProgressData `json:"progress"`
	UpdatedAtUnix int64        `json:"updated_at_unix"`
}

// OptionsData is the serializable training options.
type OptionsData struct {
	NoiseVariance   float64 `json:"noise_variance"`
	OptimizeNoise   bool    `json:"optimize_noise"`
	MaxIterations   int     `json:"max_iterations"`
	ConvergenceTol  float64 `json:"convergence_tol"`
	GradientTol     float64 `json:"gradient_tol"`
	Stability       string  `json:"stability"`
	InitialJitter   float64 `json:"initial_jitter"`
	MaxJitter       float64 `json:"max_jitter"`
	JitterFactor    float64 `json:"jitter_factor"`
	RandomSeed      int64   `json:"random_seed"`
	CheckpointEvery int     `json:"checkpoint_every,omitempty"`
}

// ProgressData is the serializable optimization progress.
type ProgressData struct {
	InitialLML     float64    `json:"initial_lml"`
	FinalLML       float64    `json:"final_lml"`
	InitialParams  []float64  `json:"initial_params"`
	FinalParams    []float64  `json:"final_params"`
	Iterations     int        `json:"iterations"`
	Converged      bool       `json:"converged"`
	Termination    string     `json:"termination_reason"`
	AppliedJitter  float64    `json:"applied_jitter"`
	CompletedSteps []StepData `json:"completed_steps"`
}

// StepData is one recorded optimizer iteration.
type StepData struct {
	Iteration  int       `json:"iteration"`
	Params     []float64 `json:"params"`
	NegLML     float64   `json:"neg_lml"`
	GradNorm   float64   `json:"grad_norm"`
	StepLength float64   `json:"step_length"`
}

// CheckpointStore persists model checkpoints atomically.
type CheckpointStore interface {
	// Save writes a complete checkpoint atomically (temp + fsync + rename).
	Save(ctx context.Context, data *ModelData) error
	// Load returns the latest complete, verified checkpoint.
	Load(ctx context.Context) (*ModelData, error)
	// Clear removes any stored checkpoint.
	Clear(ctx context.Context) error
}

// FileStore is a JSON-on-disk CheckpointStore.
//
// Durability strategy: Save serializes to a temp file in the same
// directory, fsyncs it, then atomically renames it over the checkpoint
// path and fsyncs the directory. A reader therefore either sees the
// previous complete checkpoint or the new one, never a partial write.
// Every checkpoint carries a SHA-256 checksum verified on Load.
type FileStore struct {
	dir string
}

// NewFileStore opens (and lazily creates) a directory-backed store.
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("store: directory must not be empty")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("store: cannot create %q: %w", dir, err)
	}
	return &FileStore{dir: dir}, nil
}

func (s *FileStore) path() string     { return filepath.Join(s.dir, checkpointName) }
func (s *FileStore) tempPath() string { return filepath.Join(s.dir, tempName) }

// Save writes a complete checkpoint atomically.
func (s *FileStore) Save(ctx context.Context, data *ModelData) error {
	if data == nil {
		return errors.New("store: nil checkpoint data")
	}
	if err := validateModelData(data); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	payload := encodeCanonical(data)
	sum := sha256.Sum256(payload)
	env := envelope{
		ChecksumAlgo: checksumAlgo,
		Checksum:     hex.EncodeToString(sum[:]),
		Payload:      payload,
	}
	raw := encodeCanonical(&env)

	tmp := s.tempPath()
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("store: open temp failed: %w", err)
	}
	if _, err := f.Write(raw); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("store: write temp failed: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("store: fsync temp failed: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("store: close temp failed: %w", err)
	}
	if err := os.Rename(tmp, s.path()); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("store: atomic rename failed: %w", err)
	}
	if err := syncDir(s.dir); err != nil {
		return err
	}
	return nil
}

// Load reads and verifies the latest complete checkpoint.
func (s *FileStore) Load(ctx context.Context) (*ModelData, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	raw, err := os.ReadFile(s.path())
	if errors.Is(err, os.ErrNotExist) {
		// A leftover temp file from a crash is not a usable checkpoint.
		return nil, ErrNoCheckpoint
	}
	if err != nil {
		return nil, fmt.Errorf("store: read failed: %w", err)
	}
	data, err := decodeCheckpoint(raw)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// Clear removes any stored checkpoint and stray temp file.
func (s *FileStore) Clear(ctx context.Context) error {
	for _, p := range []string{s.path(), s.tempPath()} {
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func decodeCheckpoint(raw []byte) (*ModelData, error) {
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("%w: invalid JSON envelope: %v", ErrCheckpointCorrupt, err)
	}
	if env.ChecksumAlgo != checksumAlgo {
		return nil, fmt.Errorf("%w: unsupported checksum algorithm %q", ErrCheckpointCorrupt, env.ChecksumAlgo)
	}
	sum := sha256.Sum256(env.Payload)
	if env.Checksum != hex.EncodeToString(sum[:]) {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrCheckpointCorrupt)
	}
	if !json.Valid(env.Payload) {
		return nil, fmt.Errorf("%w: payload is not valid JSON", ErrCheckpointCorrupt)
	}
	var data ModelData
	dec := json.NewDecoder(bytes.NewReader(env.Payload))
	if err := dec.Decode(&data); err != nil {
		return nil, fmt.Errorf("%w: invalid payload: %v", ErrCheckpointCorrupt, err)
	}
	if err := validateModelData(&data); err != nil {
		return nil, err
	}
	return &data, nil
}

func validateModelData(d *ModelData) error {
	if d.SchemaVersion != SchemaVersion {
		return fmt.Errorf("%w: schema version %d, want %d", ErrCheckpointCorrupt, d.SchemaVersion, SchemaVersion)
	}
	if d.ModelID == "" {
		return fmt.Errorf("%w: missing model id", ErrCheckpointCorrupt)
	}
	if d.InputDim <= 0 {
		return fmt.Errorf("%w: invalid input dimension", ErrCheckpointCorrupt)
	}
	if len(d.X) == 0 || len(d.X) != len(d.Y) {
		return fmt.Errorf("%w: x/y data missing or mismatched", ErrCheckpointCorrupt)
	}
	for i, xi := range d.X {
		if len(xi) != d.InputDim {
			return fmt.Errorf("%w: x[%d] dimension mismatch", ErrCheckpointCorrupt, i)
		}
	}
	if len(d.KernelParams) == 0 {
		return fmt.Errorf("%w: missing kernel parameters", ErrCheckpointCorrupt)
	}
	for i, p := range d.KernelParams {
		if p != p || p < 0 { // NaN or negative
			return fmt.Errorf("%w: invalid kernel parameter %d", ErrCheckpointCorrupt, i)
		}
	}
	if d.Noise != d.Noise || d.Noise < 0 {
		return fmt.Errorf("%w: invalid noise variance", ErrCheckpointCorrupt)
	}
	if d.Progress.Iterations < 0 {
		return fmt.Errorf("%w: negative iteration count", ErrCheckpointCorrupt)
	}
	return nil
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("store: open dir for fsync failed: %w", err)
	}
	defer d.Close()
	// Some filesystems do not support fsync on directories; the rename
	// itself is still atomic, so the error is advisory and ignored.
	_ = d.Sync()
	return nil
}

// encodeCanonical produces compact JSON without HTML escaping. Because
// RawMessage payloads are embedded verbatim, the bytes covered by the
// checksum are exactly the bytes present in the file.
func encodeCanonical(v any) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return bytes.TrimRight(buf.Bytes(), "\n")
}

// OptionsFromGP converts gp.Options to its serializable form.
func OptionsFromGP(o gp.Options) OptionsData {
	return OptionsData{
		NoiseVariance:   o.NoiseVariance,
		OptimizeNoise:   o.OptimizeNoise,
		MaxIterations:   o.MaxIterations,
		ConvergenceTol:  o.ConvergenceTol,
		GradientTol:     o.GradientTol,
		Stability:       string(o.Stability),
		InitialJitter:   o.InitialJitter,
		MaxJitter:       o.MaxJitter,
		JitterFactor:    o.JitterFactor,
		RandomSeed:      o.RandomSeed,
		CheckpointEvery: o.CheckpointEvery,
	}
}

// OptionsToGP converts stored options back to gp.Options.
func OptionsToGP(d OptionsData) gp.Options {
	return gp.Options{
		NoiseVariance:   d.NoiseVariance,
		OptimizeNoise:   d.OptimizeNoise,
		MaxIterations:   d.MaxIterations,
		ConvergenceTol:  d.ConvergenceTol,
		GradientTol:     d.GradientTol,
		Stability:       gp.StabilityStrategy(d.Stability),
		InitialJitter:   d.InitialJitter,
		MaxJitter:       d.MaxJitter,
		JitterFactor:    d.JitterFactor,
		RandomSeed:      d.RandomSeed,
		CheckpointEvery: d.CheckpointEvery,
	}
}

// ProgressFromGP converts gp.OptimizeProgress to its serializable form.
func ProgressFromGP(p *gp.OptimizeProgress) ProgressData {
	if p == nil {
		return ProgressData{}
	}
	d := ProgressData{
		InitialLML:    p.InitialLML,
		FinalLML:      p.FinalLML,
		InitialParams: append([]float64(nil), p.InitialParams...),
		FinalParams:   append([]float64(nil), p.FinalParams...),
		Iterations:    p.Iterations,
		Converged:     p.Converged,
		Termination:   string(p.Termination),
		AppliedJitter: p.AppliedJitter,
	}
	for _, st := range p.CompletedSteps {
		d.CompletedSteps = append(d.CompletedSteps, StepData{
			Iteration:  st.Iteration,
			Params:     append([]float64(nil), st.Params...),
			NegLML:     st.NegLML,
			GradNorm:   st.GradNorm,
			StepLength: st.StepLength,
		})
	}
	return d
}

// ProgressToGP converts stored progress back to gp.OptimizeProgress.
func ProgressToGP(d ProgressData) *gp.OptimizeProgress {
	p := &gp.OptimizeProgress{
		InitialLML:    d.InitialLML,
		FinalLML:      d.FinalLML,
		InitialParams: append([]float64(nil), d.InitialParams...),
		FinalParams:   append([]float64(nil), d.FinalParams...),
		Iterations:    d.Iterations,
		Converged:     d.Converged,
		Termination:   gp.TerminationReason(d.Termination),
		AppliedJitter: d.AppliedJitter,
	}
	for _, st := range d.CompletedSteps {
		p.CompletedSteps = append(p.CompletedSteps, gp.OptimizeStep{
			Iteration:  st.Iteration,
			Params:     append([]float64(nil), st.Params...),
			NegLML:     st.NegLML,
			GradNorm:   st.GradNorm,
			StepLength: st.StepLength,
		})
	}
	return p
}
