package gpr

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ErrCheckpointCorrupt 表示检查点内容损坏（校验失败、截断或无法解析）。
var ErrCheckpointCorrupt = errors.New("gpr: checkpoint is corrupt")

// ErrNoCheckpoint 表示存储中不存在可用检查点。
var ErrNoCheckpoint = errors.New("gpr: no checkpoint available")

// checkpointVersion 是检查点格式版本。
const checkpointVersion = 1

// Checkpoint 是一次已完整提交的训练状态。
type Checkpoint struct {
	Version    int
	ModelID    string
	KernelType KernelType
	Dim        int
	X          [][]float64
	Y          []float64
	Noise      float64
	MeanTrain  float64
	Result     OptimizeResult
	State      *optimizerState
	Completed  bool
	CreatedAt  time.Time
	Config     PersistedConfig
}

// PersistedConfig 是需要落盘的配置子集。
type PersistedConfig struct {
	Stability StabilityConfig
	Optimizer OptimizerConfig
}

// CheckpointStore 负责检查点的原子持久化与加载。
type CheckpointStore interface {
	// Save 原子写入一个完整检查点（临时文件 + fsync + rename）。
	Save(cp *Checkpoint) error
	// LoadLatest 加载最近一次完整提交的检查点。
	LoadLatest() (*Checkpoint, error)
	// LoadForModel 加载指定模型最近一次完整提交的检查点。
	LoadForModel(modelID string) (*Checkpoint, error)
	// Close 释放底层资源。
	Close() error
}

// NewFileStore 在 dir 下创建基于文件的检查点存储。
func NewFileStore(dir string) (CheckpointStore, error) {
	if dir == "" {
		return nil, errors.New("gpr: checkpoint directory must not be empty")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("gpr: create checkpoint dir: %w", err)
	}
	fs := &fileStore{dir: dir}
	if err := fs.cleanupTemp(); err != nil {
		return nil, err
	}
	return fs, nil
}

// encodeCheckpoint 将检查点序列化为带校验和的字节。
func encodeCheckpoint(cp *Checkpoint) ([]byte, error) {
	if cp == nil {
		return nil, errors.New("gpr: nil checkpoint")
	}
	wire := checkpointToWire(cp)
	payload, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("gpr: encode checkpoint: %w", err)
	}
	sum := sha256.Sum256(payload)
	var buf bytes.Buffer
	buf.Write(checkpointMagic)
	if err := binary.Write(&buf, binary.LittleEndian, uint32(checkpointVersion)); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(payload))); err != nil {
		return nil, err
	}
	buf.Write(sum[:])
	buf.Write(payload)
	return buf.Bytes(), nil
}

// decodeCheckpoint 校验并反序列化检查点；损坏时返回 ErrCheckpointCorrupt。
func decodeCheckpoint(raw []byte) (*Checkpoint, error) {
	minLen := len(checkpointMagic) + 4 + 4 + sha256.Size
	if len(raw) < minLen {
		return nil, ErrCheckpointCorrupt
	}
	if !bytes.Equal(raw[:len(checkpointMagic)], checkpointMagic) {
		return nil, ErrCheckpointCorrupt
	}
	pos := len(checkpointMagic)
	version := binary.LittleEndian.Uint32(raw[pos : pos+4])
	pos += 4
	if int(version) != checkpointVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCheckpointCorrupt, version)
	}
	size := binary.LittleEndian.Uint32(raw[pos : pos+4])
	pos += 4
	if int(size) != len(raw)-minLen {
		return nil, ErrCheckpointCorrupt
	}
	sum := raw[pos : pos+sha256.Size]
	pos += sha256.Size
	payload := raw[pos:]
	actual := sha256.Sum256(payload)
	if !bytes.Equal(sum, actual[:]) {
		return nil, ErrCheckpointCorrupt
	}
	var w checkpointWire
	if err := json.Unmarshal(payload, &w); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCheckpointCorrupt, err)
	}
	if w.Version != checkpointVersion || w.ModelID == "" {
		return nil, ErrCheckpointCorrupt
	}
	if len(w.X) != len(w.Y) || len(w.X) == 0 {
		return nil, ErrCheckpointCorrupt
	}
	cp, err := wireToCheckpoint(&w)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCheckpointCorrupt, err)
	}
	return cp, nil
}

// wireResult 是 OptimizeResult 的可序列化形式（Kernel 接口无法直接 JSON 编码）。
type wireResult struct {
	KernelType  KernelType
	Params      KernelParams
	Noise       float64
	InitialLML  float64
	FinalLML    float64
	Iterations  int
	StopReason  StopReason
	History     []IterationStat
	ElapsedNano int64
	JitterUsed  float64
}

func resultToWire(r OptimizeResult) wireResult {
	w := wireResult{
		Noise:       r.Noise,
		InitialLML:  r.InitialLML,
		FinalLML:    r.FinalLML,
		Iterations:  r.Iterations,
		StopReason:  r.StopReason,
		History:     r.History,
		ElapsedNano: int64(r.Elapsed),
		JitterUsed:  r.JitterUsed,
	}
	if r.Kernel != nil {
		w.KernelType = r.Kernel.Type()
		w.Params = r.Kernel.Params()
	}
	return w
}

func (w wireResult) toResult() (*OptimizeResult, error) {
	var k Kernel
	if w.KernelType != "" {
		var err error
		k, err = NewKernel(w.KernelType, w.Params)
		if err != nil {
			return nil, err
		}
	}
	if w.History == nil {
		w.History = []IterationStat{}
	}
	return &OptimizeResult{
		Kernel:     k,
		Noise:      w.Noise,
		InitialLML: w.InitialLML,
		FinalLML:   w.FinalLML,
		Iterations: w.Iterations,
		StopReason: w.StopReason,
		History:    w.History,
		Elapsed:    time.Duration(w.ElapsedNano),
		JitterUsed: w.JitterUsed,
	}, nil
}

// checkpointWire 是 Checkpoint 的 JSON 编码形式。
type checkpointWire struct {
	Version    int
	ModelID    string
	KernelType KernelType
	Dim        int
	X          [][]float64
	Y          []float64
	Noise      float64
	MeanTrain  float64
	Result     wireResult
	State      *optimizerState
	Completed  bool
	CreatedAt  time.Time
	Config     PersistedConfig
}

func checkpointToWire(cp *Checkpoint) checkpointWire {
	return checkpointWire{
		Version:    cp.Version,
		ModelID:    cp.ModelID,
		KernelType: cp.KernelType,
		Dim:        cp.Dim,
		X:          cp.X,
		Y:          cp.Y,
		Noise:      cp.Noise,
		MeanTrain:  cp.MeanTrain,
		Result:     resultToWire(cp.Result),
		State:      cp.State,
		Completed:  cp.Completed,
		CreatedAt:  cp.CreatedAt,
		Config:     cp.Config,
	}
}

func wireToCheckpoint(w *checkpointWire) (*Checkpoint, error) {
	res, err := w.Result.toResult()
	if err != nil {
		return nil, err
	}
	return &Checkpoint{
		Version:    w.Version,
		ModelID:    w.ModelID,
		KernelType: w.KernelType,
		Dim:        w.Dim,
		X:          w.X,
		Y:          w.Y,
		Noise:      w.Noise,
		MeanTrain:  w.MeanTrain,
		Result:     *res,
		State:      w.State,
		Completed:  w.Completed,
		CreatedAt:  w.CreatedAt,
		Config:     w.Config,
	}, nil
}

// checkpointMagic 是检查点文件的固定魔数。
var checkpointMagic = []byte("GPRCKPT\x01")

const (
	ckptSuffix = ".ckpt"
	tempSuffix = ".tmp"
	keepLatest = 3
)

type fileStore struct {
	dir string
}

func (fs *fileStore) Save(cp *Checkpoint) error {
	if cp.Version == 0 {
		cp.Version = checkpointVersion
	}
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	raw, err := encodeCheckpoint(cp)
	if err != nil {
		return err
	}
	// 写入同目录临时文件 → fsync → 原子 rename，保证读端永远看不到半写入内容。
	name := fmt.Sprintf("%s-%06d%s", cp.ModelID, checkpointSeq(cp), tempSuffix)
	tmpPath := filepath.Join(fs.dir, name)
	finalPath := filepath.Join(fs.dir, fmt.Sprintf("%s-%06d%s", cp.ModelID, checkpointSeq(cp), ckptSuffix))

	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("gpr: write checkpoint: %w", err)
	}
	if _, err := f.Write(raw); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("gpr: write checkpoint: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("gpr: fsync checkpoint: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("gpr: close checkpoint: %w", err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("gpr: commit checkpoint: %w", err)
	}
	return fs.prune(cp.ModelID)
}

func (fs *fileStore) LoadLatest() (*Checkpoint, error) {
	names, err := fs.listCheckpoints()
	if err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, ErrNoCheckpoint
	}
	return fs.loadByNames(names)
}

func (fs *fileStore) Close() error { return nil }

func (fs *fileStore) LoadForModel(modelID string) (*Checkpoint, error) {
	names, err := fs.listCheckpoints()
	if err != nil {
		return nil, err
	}
	prefix := modelID + "-"
	var mine []string
	for _, n := range names {
		if len(n) > len(prefix) && n[:len(prefix)] == prefix {
			mine = append(mine, n)
		}
	}
	if len(mine) == 0 {
		return nil, ErrNoCheckpoint
	}
	return fs.loadByNames(mine)
}

// loadByNames 按文件名从新到旧加载；首个能解析的文件必须通过校验，否则显式报错。
func (fs *fileStore) loadByNames(names []string) (*Checkpoint, error) {
	sort.Strings(names)
	for i := len(names) - 1; i >= 0; i-- {
		raw, err := os.ReadFile(filepath.Join(fs.dir, names[i]))
		if err != nil {
			continue
		}
		cp, err := decodeCheckpoint(raw)
		if err != nil {
			return nil, fmt.Errorf("gpr: checkpoint %s: %w", names[i], err)
		}
		return cp, nil
	}
	return nil, ErrNoCheckpoint
}

func (fs *fileStore) listCheckpoints() ([]string, error) {
	entries, err := os.ReadDir(fs.dir)
	if err != nil {
		return nil, fmt.Errorf("gpr: read checkpoint dir: %w", err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ckptSuffix {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names, nil
}

func (fs *fileStore) cleanupTemp() error {
	entries, err := os.ReadDir(fs.dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == tempSuffix {
			if err := os.Remove(filepath.Join(fs.dir, e.Name())); err != nil {
				return err
			}
		}
	}
	return nil
}

// prune 仅保留同一模型最近 keepLatest 份检查点。
func (fs *fileStore) prune(modelID string) error {
	names, err := fs.listCheckpoints()
	if err != nil {
		return err
	}
	prefix := modelID + "-"
	var mine []string
	for _, n := range names {
		if len(n) > len(prefix) && n[:len(prefix)] == prefix {
			mine = append(mine, n)
		}
	}
	sort.Strings(mine)
	for len(mine) > keepLatest {
		if err := os.Remove(filepath.Join(fs.dir, mine[0])); err != nil {
			return err
		}
		mine = mine[1:]
	}
	return nil
}

// checkpointSeq 用迭代次数生成检查点序号；完成态使用高位保证排在最后。
func checkpointSeq(cp *Checkpoint) int {
	seq := cp.Result.Iterations
	if cp.Completed {
		seq += 1_000_000
	}
	return seq
}
