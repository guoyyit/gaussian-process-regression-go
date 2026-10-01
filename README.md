# Gaussian Process Regression (Go)

一个零外部依赖、可本地运行的高斯过程回归（GPR）服务。支持单维/多维输入、
均值与方差预测、多种核函数、基于对数边际似然（LML）的超参数自动优化、
病态输入的可配置稳定性策略、原子检查点与崩溃续训、无锁并发版本管理，
以及与全量重训逐位一致的增量追加。

## 快速开始

```bash
# 运行内置演示：合成带噪声 sin 数据 → 训练 → 追加 → 并发查询
go run ./cmd/gpr-server -dir .gpr-data -kernel rbf -iters 100 -points 40 -seed 42

# 再次运行会从最近一次完整检查点恢复，而不是重新训练
go run ./cmd/gpr-server -dir .gpr-data

# 运行全部测试（含竞态检测）
go test -race -count=1 ./...
```

作为库使用：

```go
svc, _ := gpr.NewService(gpr.ServiceConfig{CheckpointDir: ".gpr-data", Seed: 42})
defer svc.Close()

cfg := gpr.TrainConfig{
    KernelType:    gpr.KernelRBF,
    InitialParams: gpr.KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1},
    Noise:         0.1,      // 观测噪声标准差 σ
    Optimize:      true,
    Stability:     gpr.DefaultStabilityConfig(),
}
cfg.Optimizer = gpr.OptimizerConfig{MaxIterations: 100, Seed: 42}

res, _ := svc.Train(ctx, xTrain, yTrain, cfg)
p, _ := svc.Predict([]float64{1.5})
fmt.Println(p.Mean, p.Variance)

res2, _ := svc.Append(ctx, [][]float64{{3.1}}, []float64{0.04})
```

## 数学模型与假设

观测模型为

```
y = f(X) + ε,   ε ~ N(0, σ² I),   f ~ GP(0, k_θ(x, x'))
```

- 使用**零均值 + 常数均值中心化**：训练前对目标取样本均值 `m = mean(y)`，
  对 `y - m` 做标准零均值 GP 推断，预测均值再加回 `m`。这等价于常数均值函数。
- 给定训练集 `X (n×d)`、`y (n)`，记 `K_y = K_θ(X,X) + σ² I`，
  `k* = K_θ(X, x*)`，`k** = k_θ(x*, x*)`：
  - 后验均值：`μ* = m + k*ᵀ K_y⁻¹ (y - m)`
  - 后验方差：`ν* = k** - k*ᵀ K_y⁻¹ k*`
  - **对数边际似然**：`log p(y|X,θ,σ) = -½ (y-m)ᵀ K_y⁻¹ (y-m) - ½ log|K_y| - n/2 log(2π)`

这里的方差是**潜在函数 f\* 的后验（条件）方差**。因此在观测点处
`ν* ≈ σ²`（噪声水平）而不是 0；远离观测点时 `ν* → k(x*,x*)`（先验方差）。

### 核函数

| 类型 | 公式（`r = ‖a-b‖₂`） | 可调参数 |
| --- | --- | --- |
| `rbf`（平方指数） | `sf² exp(-r²/(2ℓ²))` | ℓ, sf² |
| `matern32` | `sf² (1+√3 r/ℓ) exp(-√3 r/ℓ)` | ℓ, sf² |
| `linear`（归一化线性） | `sf² (a·b)/ℓ²` | ℓ, sf² |
| `rational_quadratic` | `sf² (1 + r²/(2αℓ²))^(-α)` | ℓ, sf², α |

- `LengthScale` ℓ：特征长度尺度，越大函数越平滑、相关范围越远。
- `SignalVariance` sf²：先验信号方差（核对角元量级）。
- `Alpha` α：仅 RQ 使用，尺度混合参数（α→∞ 时 RQ 趋近 RBF）。
- `Noise`：观测噪声的**标准差** σ（内部使用 σ²）。

## 超参数优化

- 参数向量取对数空间（保证正数、消除尺度差异）：
  `φ = [log ℓ, log sf², (log α), log σ]`，RQ 为 4 维，其余为 3 维。
- 优化器为 **Adam（梯度上升）**，β₁=0.9、β₂=0.999、ε=1e-8，学习率默认 0.05。
- 梯度用**中心差分**计算：`∂LML/∂φᵢ ≈ (LML(φᵢ+h)-LML(φᵢ-h))/(2h)`，
  默认 `h = 1e-5`。
- 收敛判据与终止原因（`OptimizeResult.StopReason`）：
  - `converged`：本迭代所有参数的 Adam 步长最大值 < `Tolerance`（默认 1e-6）。
  - `no_improvement`：连续 `NoImproveWindow`（默认 10）次迭代 LML 没有提升。
  - `max_iterations`：达到 `MaxIterations`（默认 200）。
  - `numeric_failure`：当前参数下 LML/梯度无法可靠计算（病态到抖动上限仍失败）。
  - `optimization_disabled`：`TrainConfig.Optimize=false`，不优化。
- **单调保证**：最终返回的参数是整个轨迹上评估过 LML 的最佳点；
  若优化结束时的最佳点仍差于初始点，则回退到初始参数。
  因此 `FinalLML >= InitialLML`（数值容差 1e-8）。
- 参数会被裁剪到 `log ∈ [-8, 8]`，噪声标准差下限 `1e-8`，防止溢出/退化。

## 数值稳定性策略

核矩阵在输入点重合或过于接近时会奇异。策略由 `StabilityConfig.Policy` 配置：

| 策略 | 行为 |
| --- | --- |
| `JitterOff` | 不加抖动；Cholesky 失败即返回 `gpr.ErrSingular`（可识别错误，绝不返回 NaN/Inf）。 |
| `JitterFixed` | 给对角元一次性加 `InitialJitter`，失败即 `ErrSingular`。 |
| `JitterAdaptive`（默认） | 先无抖动尝试；失败后按 10× 递增（1e-10 → … → `MaxJitter`，默认上限 1e-2），任一量级成功即返回。 |

其他保证：

- 分解前对矩阵做对称化 `(A+Aᵀ)/2`，消除舍入造成的非对称。
- 正定判据为 Cholesky 主元 `> 1e-10 · max|diag|`。
- 预测方差先裁剪到 `MinVariance`（默认 0），再对 `[-1e-8, 0)` 的负零误差钳为 0；
  仍为负或出现 NaN/Inf 时返回 `ErrSingular`，不静默给出错误结果。
- 所有对外返回的均值/方差都经过有限性检查。

## 持久化、检查点与崩溃续训

- 存储目录由 `ServiceConfig.CheckpointDir` 指定；为空则禁用持久化（纯内存）。
- 检查点文件：`<modelID>-<seq>.ckpt`，模型 ID 是训练数据（维度、数量、
  输入与目标的逐位浮点表示）的 SHA-256 前 16 字节，因此**相同数据复用同一检查点链**。
- 写入协议保证读者永远看不到半写入内容：
  1. 写同目录临时文件 `<name>.tmp`；
  2. `fsync` 落盘；
  3. `rename(2)` 原子替换为最终文件名；
  4. 启动时清理残留 `.tmp`。
- 文件格式：8 字节魔数 `GPRCKPT\x01` + 版本号(u32) + 载荷长度(u32)
  + SHA-256(载荷) + JSON 载荷。魔数不符、长度不符、截断或校验和不匹配
  一律返回 `gpr.ErrCheckpointCorrupt` 并**拒绝加载**，不会静默忽略。
- **续训不重复迭代**：每 `CheckpointEvery`（默认 10）次迭代，优化器状态
  （迭代号、参数向量、Adam 的 m/v、当前最佳 LML）作为未完成检查点原子落盘。
  崩溃后再次 `Train` 相同数据时从 `state.Iteration+1` 继续，
  已完成的迭代不会再次执行；恢复路径与一次性训练使用相同的确定性更新，
  最终参数/LML 与未中断运行逐位一致（见 `TestCrashResumeDoesNotReapplyIters`）。
- 训练完成后原子写入 `Completed=true` 的检查点，序号加 1e6 保证排在最前。
  每个模型仅保留最近 3 份检查点。

## 并发模型

- `Service` 内部用 `atomic.Pointer[modelSnapshot]` 保存当前版本；
  快照是**不可变**的（数据、核、`K_y` 因子、α 向量全部只读）。
- 更新（`Train`/`Append`）在私有副本上构建并优化完整新模型，
  完成后单次原子指针替换发布；旧版本读者继续在旧快照上完成查询。
- 因此查询**永不阻塞**，且任何时刻读到的都是某一个**完整已提交**版本，
  不可能出现新旧核参数/训练状态混合（`go test -race` 覆盖）。
- 多个更新之间用互斥锁串行化，保证并发 `Append` 不会丢观测点；
  该锁只在更新路径持有，读路径完全无锁。
- 在第一个版本提交之前调用 `Predict`/`Append` 返回显式错误。

## 增量追加

- `Append(x, y)` 校验维度后，把新观测拼到当前数据集上，
  **从同一组初始超参数（ℓ=1, sf²=1, α=1, σ=0.01）冷启动**，
  使用相同的种子、迭代上限与优化器配置重新优化。
- 因为优化器是确定性的（数值梯度 + Adam，无任何随机源），
  追加后结果与“用全部数据一次性 `Train`”**逐位一致**，
  而不仅是浮点误差内的近似；见 `TestAppendMatchesFullRetrain`。
- 若采用不同初始参数或不同迭代数，则结果会有差异，这属于预期行为。

## 确定性与复现

- 训练/优化路径不使用 `math/rand` 全局状态或 goroutine 调度相关时序；
  Adam 与中心差分在相同输入下逐位确定。
- `ServiceConfig.Seed` 与 `OptimizerConfig.Seed` 当前用于标识复现配方
  （及演示程序的合成数据 LCG）。相同数据 + 相同配置 + 相同种子
  ⇒ 相同超参数、LML 轨迹与预测，跨全新进程也成立。
- **浮点比较规则**（测试中）：
  - 确定性场景（相同配方）使用**精确相等** `==` 比较浮点参数、LML、均值、方差；
  - 数值正确性场景使用容差比较（观测点均值误差 `1e-4`、方差与 σ² 误差 `1e-4`，
    核函数值误差 `1e-12`），容差来源于 IEEE-754 双精度与 Cholesky 分解的
    O(nε) 舍入累积；
  - 平台相关的浮点累加顺序在本实现中固定（不依赖并行归约），同一架构/编译器下可复现。

## 复杂度

设 n 为训练点数、d 为输入维度、T 为优化迭代数、p 为超参数维度（3 或 4）：

| 操作 | 时间 | 额外空间 |
| --- | --- | --- |
| 构建核矩阵 | `O(n² d)` | `O(n²)` |
| Cholesky 分解 | `O(n³)` | `O(n²)` |
| 单次训练（固定超参） | `O(n³ + n² d)` | `O(n²)` |
| 单次 LML 评估 | `O(n³ + n² d)` | `O(n²)` |
| 超参数优化 | `O(T · p · n³)`（中心差分 2p 次评估/迭代） | `O(n²)` |
| 单点/批量预测(q 点) | `O(n² + q n²)`（复用已存 L 因子） | `O(n)` |
| 检查点写入 | `O(n²)`（核参数与数据序列化，不含 n² 矩阵） | 磁盘 `O(nd)` |

检查点只保存数据与超参数，不保存 n×n 矩阵；恢复时重新做一次 `O(n³)` 分解。

## 包结构

| 文件 | 职责 |
| --- | --- |
| `data.go` | 数据集构造、维度/有限性校验、追加 |
| `kernel.go` | 四种核函数与参数校验 |
| `linalg.go` | Cholesky 分解、前代/回代、抖动策略 |
| `gp.go` | 模型拟合、均值/方差预测、核矩阵 |
| `optimize.go` | LML、数值梯度、Adam 优化、续训状态 |
| `checkpoint.go` | 原子检查点存储、校验和、损坏检测 |
| `service.go` | 并发版本管理、训练/追加/恢复编排 |
| `cmd/gpr-server` | 本地可运行演示（含并发查询演示） |
