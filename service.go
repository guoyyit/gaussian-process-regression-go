package gpr

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

// ServiceConfig 配置服务级行为。
type ServiceConfig struct {
	CheckpointDir string
	Seed          int64
}

// VersionInfo 描述一个已提交的模型版本。
type VersionInfo struct {
	ModelID    string
	Iteration  int
	Completed  bool
	FinalLML   float64
	StopReason StopReason
}

// modelSnapshot 是不可变的模型版本，查询直接读快照。
type modelSnapshot struct {
	info  VersionInfo
	model *Model
	cp    *Checkpoint
}

// Service 是并发安全的 GP 回归服务。
type Service struct {
	cfg      ServiceConfig
	store    CheckpointStore
	current  atomic.Pointer[modelSnapshot]
	updateMu sync.Mutex
}

// NewService 创建服务；dir 为空表示禁用持久化。
func NewService(cfg ServiceConfig) (*Service, error) {
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	s := &Service{cfg: cfg}
	if cfg.CheckpointDir != "" {
		store, err := NewFileStore(cfg.CheckpointDir)
		if err != nil {
			return nil, err
		}
		s.store = store
	}
	return s, nil
}

// Close 关闭服务及其存储。
func (s *Service) Close() error {
	if s.store != nil {
		return s.store.Close()
	}
	return nil
}

// Train 执行一次训练；若存储中存在该数据的未完成检查点则自动续训。
func (s *Service) Train(ctx context.Context, x [][]float64, y []float64, cfg TrainConfig) (*OptimizeResult, error) {
	data, err := NewDataset(x, y)
	if err != nil {
		return nil, err
	}
	return s.trainData(ctx, data, cfg, modelID(data))
}

// Predict 在当前已提交版本上预测。
func (s *Service) Predict(x []float64) (Prediction, error) {
	snap := s.current.Load()
	if snap == nil {
		return Prediction{}, errors.New("gpr: no trained model available")
	}
	return snap.model.Predict(x)
}

// PredictBatch 在当前已提交版本上批量预测。
func (s *Service) PredictBatch(x [][]float64) ([]Prediction, error) {
	snap := s.current.Load()
	if snap == nil {
		return nil, errors.New("gpr: no trained model available")
	}
	return snap.model.PredictBatch(x)
}

// Append 增量追加观测点并产出新版本；预测语义等价于全量重训。
func (s *Service) Append(ctx context.Context, x [][]float64, y []float64) (*OptimizeResult, error) {
	snap := s.current.Load()
	if snap == nil {
		return nil, errors.New("gpr: no existing model to append to; call Train first")
	}
	data, err := snap.model.data.Append(x, y)
	if err != nil {
		return nil, err
	}
	// 从同一组初始超参数冷启动，在相同数据与种子下逐位等价于全量重训。
	cfg := trainConfigFromCheckpoint(snap.cp)
	return s.trainData(ctx, data, cfg, modelID(data))
}

// CurrentVersion 返回当前已提交版本的信息；无模型时 ok=false。
func (s *Service) CurrentVersion() (VersionInfo, bool) {
	snap := s.current.Load()
	if snap == nil {
		return VersionInfo{}, false
	}
	return snap.info, true
}

// Restore 从存储中的最近检查点恢复当前版本。
func (s *Service) Restore() error {
	if s.store == nil {
		return errors.New("gpr: persistence disabled")
	}
	cp, err := s.store.LoadLatest()
	if err != nil {
		return err
	}
	snap, err := snapshotFromCheckpoint(cp)
	if err != nil {
		return err
	}
	s.current.Store(snap)
	return nil
}

// ErrNoModel 表示服务上还没有任何已提交模型。
var ErrNoModel = errors.New("gpr: no trained model available")

func (s *Service) trainData(ctx context.Context, data *Dataset, cfg TrainConfig, id string) (*OptimizeResult, error) {
	// 更新串行化：保证并发 Append 不丢观测点；查询走原子快照，不持此锁、不被阻塞。
	s.updateMu.Lock()
	defer s.updateMu.Unlock()

	var err error
	cfg, err = validateTrainConfig(cfg, data.Dim())
	if err != nil {
		return nil, err
	}

	// 续训：加载该数据模型最近一次完整检查点，跳过已完成迭代。
	var resume *optimizerState
	if s.store != nil {
		if cp, lerr := s.store.LoadForModel(id); lerr == nil && !cp.Completed {
			resume = cp.State
		} else if lerr != nil && !errors.Is(lerr, ErrNoCheckpoint) {
			return nil, lerr
		}
	}

	persisted := PersistedConfig{Stability: cfg.Stability, Optimizer: cfg.Optimizer}

	initialPacked := packParams(mustKernel(cfg.KernelType, cfg.InitialParams), cfg.Noise)
	initialLML, _, ierr := evaluateLML(data, cfg.KernelType, initialPacked, cfg.Stability)
	if ierr != nil {
		return nil, ierr
	}

	// 每个检查点边界原子落盘：训练中途崩溃只会丢掉最后一个未提交检查点。
	userCallback := cfg.onCheckpoint
	cfg.onCheckpoint = func(iteration int, lml float64, params, mom, vel []float64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if s.store != nil {
			r := interimResult(cfg, params, iteration, lml, initialLML)
			cp := s.buildCheckpoint(id, data, cfg, r, &optimizerState{
				Iteration: iteration,
				LML:       lml,
				Params:    append([]float64(nil), params...),
				M:         append([]float64(nil), mom...),
				V:         append([]float64(nil), vel...),
			}, false, persisted)
			if err := s.store.Save(cp); err != nil {
				return err
			}
		}
		if userCallback != nil {
			return userCallback(iteration, lml, params, mom, vel)
		}
		return nil
	}

	final, _, err := runOptimizer(data, cfg, resume)
	if err != nil {
		return nil, err
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// 用最终参数构建不可变模型并原子发布版本。
	model, err := fitGP(data, final.Kernel, final.Noise, meanValue(data.Y), cfg.Stability)
	if err != nil {
		return nil, err
	}
	// 完成态检查点不携带中间优化器状态：重启后直接发布该版本，无需续训。
	cp := s.buildCheckpoint(id, data, cfg, final, nil, true, persisted)
	if s.store != nil {
		if err := s.store.Save(cp); err != nil {
			return nil, err
		}
	}
	snap := &modelSnapshot{
		info: VersionInfo{
			ModelID:    id,
			Iteration:  final.Iterations,
			Completed:  true,
			FinalLML:   final.FinalLML,
			StopReason: final.StopReason,
		},
		model: model,
		cp:    cp,
	}
	s.current.Store(snap)
	return final, nil
}

func (s *Service) buildCheckpoint(id string, data *Dataset, cfg TrainConfig,
	r *OptimizeResult, state *optimizerState, completed bool, persisted PersistedConfig) *Checkpoint {
	return &Checkpoint{
		Version:    checkpointVersion,
		ModelID:    id,
		KernelType: cfg.KernelType,
		Dim:        data.Dim(),
		X:          data.X,
		Y:          data.Y,
		Noise:      r.Noise,
		MeanTrain:  meanValue(data.Y),
		Result:     *r,
		State:      state,
		Completed:  completed,
		Config:     persisted,
	}
}

func snapshotFromCheckpoint(cp *Checkpoint) (*modelSnapshot, error) {
	data, err := NewDataset(cp.X, cp.Y)
	if err != nil {
		return nil, err
	}
	kernel := cp.Result.Kernel
	if kernel == nil {
		return nil, ErrCheckpointCorrupt
	}
	model, err := fitGP(data, kernel, cp.Noise, cp.MeanTrain, cp.Config.Stability)
	if err != nil {
		return nil, err
	}
	return &modelSnapshot{
		info: VersionInfo{
			ModelID:    cp.ModelID,
			Iteration:  cp.Result.Iterations,
			Completed:  cp.Completed,
			FinalLML:   cp.Result.FinalLML,
			StopReason: cp.Result.StopReason,
		},
		model: model,
		cp:    cp,
	}, nil
}

func trainConfigFromCheckpoint(cp *Checkpoint) TrainConfig {
	return TrainConfig{
		KernelType:    cp.KernelType,
		InitialParams: initialParamsFor(cp.KernelType),
		Noise:         initialNoiseFor(cp.KernelType),
		Optimize:      true,
		Stability:     cp.Config.Stability,
		Optimizer:     cp.Config.Optimizer,
	}
}

// interimResult 构造优化中途检查点的结果快照。
func interimResult(cfg TrainConfig, packed []float64, iteration int, lml, initialLML float64) *OptimizeResult {
	p, noise, _ := unpackParams(cfg.KernelType, packed)
	k, _ := NewKernel(cfg.KernelType, p)
	return &OptimizeResult{
		Kernel:     k,
		Noise:      noise,
		InitialLML: initialLML,
		FinalLML:   lml,
		Iterations: iteration,
		StopReason: StopReason("running"),
		History:    []IterationStat{statFromPacked(iteration, lml, cfg.KernelType, packed)},
	}
}

func initialParamsFor(t KernelType) KernelParams {
	return KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}
}

func initialNoiseFor(_ KernelType) float64 {
	return defaultTrainConfig().Noise
}

// modelID 基于训练数据内容生成稳定标识；相同数据得到相同 ID（与数据顺序相关）。
func modelID(data *Dataset) string {
	h := sha256.New()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(data.Dim()))
	h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], uint64(data.Len()))
	h.Write(buf[:])
	for i := range data.X {
		for _, v := range data.X[i] {
			bits := math.Float64bits(v)
			binary.LittleEndian.PutUint64(buf[:], bits)
			h.Write(buf[:])
		}
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(data.Y[i]))
		h.Write(buf[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))[:16]
}
