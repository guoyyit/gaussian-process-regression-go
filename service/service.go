package service

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/guoyyit/gaussian-process-regression-go/gp"
	"github.com/guoyyit/gaussian-process-regression-go/kernel"
	"github.com/guoyyit/gaussian-process-regression-go/store"
)

// TrainRequest configures a new training run.
type TrainRequest struct {
	ModelID         string
	Kernel          kernel.Config
	X               [][]float64
	Y               []float64
	Options         gp.Options
	Resume          bool
	CheckpointEvery int
}

// VersionedModel is an immutable snapshot of a committed model version.
type VersionedModel struct {
	Version int64
	Model   *gp.Model
}

var (
	errModelNotFound = errors.New("service: model not found")
	errModelExists   = errors.New("service: model already exists")
)

// modelEntry holds an atomic pointer to the latest committed version.
// Readers load the pointer without holding the registry lock, so queries
// are never blocked by an in-progress update and always observe either the
// previous complete version or the new complete version, never a mix.
type modelEntry struct {
	current atomic.Pointer[VersionedModel]

	mu        sync.Mutex // serializes updates for this single model
	kcfg      kernel.Config
	opts      gp.Options
	modelID   string
	version   int64
	initialKP []float64
}

// Service coordinates training, prediction, persistence and concurrency.
type Service struct {
	storeFactory func(modelID string) (store.CheckpointStore, error)

	mu     sync.RWMutex
	models map[string]*modelEntry
}

// New constructs a Service backed by the given checkpoint store. The same
// store is used for every model id (single-model file layout).
func New(st store.CheckpointStore) *Service {
	return &Service{
		storeFactory: func(string) (store.CheckpointStore, error) { return st, nil },
		models:       make(map[string]*modelEntry),
	}
}

// NewMultiModel constructs a Service that keeps one checkpoint file per
// model id under baseDir.
func NewMultiModel(baseDir string) (*Service, error) {
	if baseDir == "" {
		return nil, errors.New("service: base directory must not be empty")
	}
	return &Service{
		storeFactory: func(id string) (store.CheckpointStore, error) {
			return store.NewFileStore(filepath.Join(baseDir, id))
		},
		models: make(map[string]*modelEntry),
	}, nil
}

// Train trains a model from scratch, checkpointing as it goes, then
// publishes a new immutable version.
func (s *Service) Train(ctx context.Context, req TrainRequest) (*VersionedModel, error) {
	if req.ModelID == "" {
		return nil, errors.New("service: ModelID required")
	}
	entry, err := s.obtainEntry(req, true)
	if err != nil {
		return nil, err
	}
	entry.mu.Lock()
	defer entry.mu.Unlock()

	st, err := s.storeFactory(req.ModelID)
	if err != nil {
		return nil, err
	}

	opts := req.Options
	opts.CheckpointEvery = req.CheckpointEvery

	k, err := kernel.New(req.Kernel)
	if err != nil {
		return nil, err
	}

	m, err := gp.NewModel(k, cloneX(req.X), append([]float64(nil), req.Y...), opts)
	if err != nil {
		return nil, err
	}

	attachCheckpointHook(m, st, entry, req.X, req.Y, req.Kernel)

	var progress *gp.OptimizeProgress
	if req.Resume {
		data, lerr := st.Load(ctx)
		if lerr == nil {
			progress = store.ProgressToGP(data.Progress)
		} else if !errors.Is(lerr, store.ErrNoCheckpoint) {
			return nil, lerr
		}
	}
	if progress != nil {
		progress, err = m.Resume(*progress)
	} else {
		progress, err = m.Train()
	}
	if err != nil {
		return nil, err
	}
	if err := m.Prepare(); err != nil {
		return nil, err
	}

	if err := saveFinal(ctx, st, entry, m, req.X, req.Y); err != nil {
		return nil, err
	}

	entry.kcfg = req.Kernel
	entry.opts = opts
	return s.publish(entry, m), nil
}

// AppendObservations incrementally adds observations and retrains on the
// full merged dataset from the same initial hyperparameters, producing a
// result equivalent (within documented FP tolerance) to training one model
// on all data at once.
func (s *Service) AppendObservations(ctx context.Context, modelID string, xNew [][]float64, yNew []float64, opts gp.Options) (*VersionedModel, error) {
	entry := s.lookup(modelID)
	if entry == nil {
		return nil, fmt.Errorf("%w: %s", errModelNotFound, modelID)
	}
	entry.mu.Lock()
	defer entry.mu.Unlock()

	cur := entry.current.Load()
	if cur == nil {
		return nil, fmt.Errorf("%w: %s (not trained yet)", errModelNotFound, modelID)
	}
	if len(xNew) != len(yNew) || len(xNew) == 0 {
		return nil, errors.New("service: append requires equal, non-empty xNew/yNew")
	}

	prev := cur.Model
	mergedX := append(cloneX(prev.X), cloneX(xNew)...)
	mergedY := make([]float64, 0, len(prev.Y)+len(yNew))
	mergedY = append(mergedY, prev.Y...)
	mergedY = append(mergedY, yNew...)

	// Rebuild a fresh kernel with the SAME initial parameters as the
	// original training, so append + retrain is deterministic and matches
	// a from-scratch run on the merged data.
	kc := entry.kcfg
	k, err := kernel.New(kc)
	if err != nil {
		return nil, err
	}
	m, err := gp.NewModel(k, mergedX, mergedY, opts)
	if err != nil {
		return nil, err
	}
	st, err := s.storeFactory(modelID)
	if err != nil {
		return nil, err
	}
	attachCheckpointHook(m, st, entry, mergedX, mergedY, kc)
	if _, err := m.Train(); err != nil {
		return nil, err
	}
	if err := m.Prepare(); err != nil {
		return nil, err
	}
	if err := saveFinal(ctx, st, entry, m, mergedX, mergedY); err != nil {
		return nil, err
	}
	return s.publish(entry, m), nil
}

// Current returns the latest committed model version.
func (s *Service) Current(modelID string) (*VersionedModel, error) {
	entry := s.lookup(modelID)
	if entry == nil {
		return nil, fmt.Errorf("%w: %s", errModelNotFound, modelID)
	}
	v := entry.current.Load()
	if v == nil {
		return nil, fmt.Errorf("%w: %s (not trained yet)", errModelNotFound, modelID)
	}
	return v, nil
}

// Predict runs a read-only prediction against the latest committed model.
// It never blocks on in-progress updates.
func (s *Service) Predict(modelID string, x []float64) (gp.Prediction, error) {
	v, err := s.Current(modelID)
	if err != nil {
		return gp.Prediction{}, err
	}
	return v.Model.Predict(x)
}

// PredictBatch predicts multiple points.
func (s *Service) PredictBatch(modelID string, xs [][]float64) ([]gp.Prediction, error) {
	v, err := s.Current(modelID)
	if err != nil {
		return nil, err
	}
	return v.Model.PredictBatch(xs)
}

// Restore loads the latest checkpoint, rebuilding the model and
// republishing it without rerunning completed iterations. If the
// optimization was unfinished it resumes from the last checkpoint.
func (s *Service) Restore(ctx context.Context, modelID string, kcfg kernel.Config, resume bool) (*VersionedModel, error) {
	entry := s.lookupOrCreate(modelID, kcfg)
	entry.mu.Lock()
	defer entry.mu.Unlock()

	st, err := s.storeFactory(modelID)
	if err != nil {
		return nil, err
	}
	data, err := st.Load(ctx)
	if err != nil {
		return nil, err
	}

	kc := kernelConfigFromData(data)
	k, err := kernel.New(kc)
	if err != nil {
		return nil, err
	}
	nk := k.NumParams()
	if len(data.KernelParams) != nk {
		return nil, fmt.Errorf("%w: stored kernel params %d != expected %d", store.ErrCheckpointCorrupt, len(data.KernelParams), nk)
	}
	if err := k.SetParams(append([]float64(nil), data.KernelParams...)); err != nil {
		return nil, err
	}
	opts := store.OptionsToGP(data.Options)
	opts.NoiseVariance = data.Noise
	m, err := gp.NewModel(k, cloneX(data.X), append([]float64(nil), data.Y...), opts)
	if err != nil {
		return nil, err
	}
	m.Noise = data.Noise
	if progress := store.ProgressToGP(data.Progress); progress.Iterations > 0 {
		m.SetProgress(*progress)
	}

	if resume && !data.Progress.Converged {
		progress := store.ProgressToGP(data.Progress)
		attachCheckpointHook(m, st, entry, data.X, data.Y, kc)
		if _, err := m.Resume(*progress); err != nil {
			return nil, err
		}
	}
	if err := m.Prepare(); err != nil {
		return nil, err
	}
	if resume {
		if err := saveFinal(ctx, st, entry, m, data.X, data.Y); err != nil {
			return nil, err
		}
	}
	entry.kcfg = kc
	entry.opts = opts
	return s.publish(entry, m), nil
}

func (s *Service) obtainEntry(req TrainRequest, create bool) (*modelEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.models[req.ModelID]; ok {
		if create && e.current.Load() != nil {
			// Allow re-train via explicit call? Reject to keep versions.
			return nil, fmt.Errorf("%w: %s", errModelExists, req.ModelID)
		}
		return e, nil
	}
	if !create {
		return nil, fmt.Errorf("%w: %s", errModelNotFound, req.ModelID)
	}
	e := &modelEntry{
		modelID: req.ModelID,
		kcfg:    req.Kernel,
		opts:    req.Options,
	}
	s.models[req.ModelID] = e
	return e, nil
}

func (s *Service) lookup(id string) *modelEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.models[id]
}

// lookupOrCreate returns the existing entry, or registers a new one.
// Used by Restore, which is the legitimate entry point for a fresh process.
func (s *Service) lookupOrCreate(id string, kcfg kernel.Config) *modelEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.models[id]; ok {
		return e
	}
	e := &modelEntry{modelID: id, kcfg: kcfg}
	s.models[id] = e
	return e
}

func (s *Service) publish(entry *modelEntry, m *gp.Model) *VersionedModel {
	v := entry.version + 1
	entry.version = v
	vm := &VersionedModel{Version: v, Model: m}
	entry.current.Store(vm)
	return vm
}

func saveFinal(ctx context.Context, st store.CheckpointStore, entry *modelEntry, m *gp.Model, x [][]float64, y []float64) error {
	data := buildModelData(entry, m, x, y)
	return st.Save(ctx, data)
}

func buildModelData(entry *modelEntry, m *gp.Model, x [][]float64, y []float64) *store.ModelData {
	data := &store.ModelData{
		SchemaVersion: store.SchemaVersion,
		ModelID:       entry.modelID,
		KernelType:    m.Kernel.Name(),
		InputDim:      m.Kernel.InputDim(),
		KernelParams:  m.Kernel.Params(),
		Noise:         m.Noise,
		X:             cloneX(x),
		Y:             append([]float64(nil), y...),
		Options:       store.OptionsFromGP(m.Options),
		UpdatedAtUnix: time.Now().Unix(),
	}
	if p := m.Progress(); p != nil {
		data.Progress = store.ProgressFromGP(p)
	}
	return data
}

func attachCheckpointHook(m *gp.Model, st store.CheckpointStore, entry *modelEntry, x [][]float64, y []float64, kcfg kernel.Config) {
	m.Options.CheckpointHook = func(p *gp.OptimizeProgress) {
		// Hook fires synchronously during optimization; the service mutex
		// for this model is already held by the training goroutine. Save
		// uses a background context as cancellation is checked by Train.
		snapshot := buildModelData(entry, m, x, y)
		snapshot.Progress = store.ProgressFromGP(p)
		_ = st.Save(context.Background(), snapshot)
	}
}

func cloneX(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = append([]float64(nil), x[i]...)
	}
	return out
}

// kernelConfigFromData reconstructs a kernel.Config that matches the
// parameter layout recorded in a checkpoint. Defaults for zero-valued
// scale terms are applied by kernel.New on construction; concrete stored
// parameters are reapplied via SetParams.
func kernelConfigFromData(data *store.ModelData) kernel.Config {
	return kernel.Config{
		Type:     kernel.Type(data.KernelType),
		InputDim: data.InputDim,
	}
}
