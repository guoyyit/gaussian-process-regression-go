# Gaussian Process Regression Service (Go)

A fully local Gaussian process regression (GPR) library and service. It
learns a continuous latent function from noisy observations and predicts a
mean together with an uncertainty (variance). The full lifecycle is
covered: training, prediction, persistence, crash-safe incremental
optimization, incremental observation appending and concurrent use. No
external services are required.

## Features

- 1-D and multi-D inputs; every prediction returns mean, latent variance
  and noisy-observation variance; variances are never negative.
- Kernels: squared-exponential (`rbf`), Matern-3/2, Matern-5/2 (with
  scalar or ARD per-dimension length scales) and dot-product `linear`.
- Hyperparameter optimization by maximizing the log marginal likelihood
  (LML) with L-BFGS and analytic gradients.
- Optimization reports a convergence criterion, iteration cap and explicit
  termination reason. The final LML is never lower than the initial LML.
- Configurable numerical stability policy (`jitter` or `strict`) for
  near-duplicate points and ill-conditioned kernel matrices; NaN/Inf are
  never returned silently.
- Atomic, checksummed checkpoints; a crash mid-training resumes from the
  last complete checkpoint without re-applying finished iterations, and a
  half-written or corrupted checkpoint is refused with an explicit error.
- Lock-free versioned reads: queries always see one complete committed
  model version and are not blocked by updates.
- Incremental append is equivalent (within a documented FP bound) to
  retraining on all data at once.
- Deterministic: fixed seeds and exact-reproducible results.

## Quick start

```sh
# End-to-end local demo: generate data, train, predict, checkpoint, restore.
go run ./cmd/gprs demo -dir .gprs-demo -n 40 -kernel rbf -iters 60

# Serve a trained model over HTTP (predicts after a checkpoint exists).
go run ./cmd/gprs serve -addr 127.0.0.1:8080 -dir .gprs-demo
curl -s 127.0.0.1:8080/predict -d '{"x":[2.0]}'
# {"mean":0.461...,"observed_variance":...,"variance":...}
```

Library usage:

```go
k, _ := kernel.New(kernel.Config{Type: kernel.RBF, InputDim: 1, LengthScales: []float64{2}})
opts := gp.DefaultOptions()
opts.NoiseVariance = 0.02
opts.OptimizeNoise = true
m, _ := gp.NewModel(k, x, y, opts)
prog, _ := m.Train()
p, _ := m.Predict([]float64{2.0})
fmt.Println(p.Mean, p.Variance, p.ObservedVariance, prog.Termination)
```

## Mathematical model

Assumptions:

- Observations satisfy `y_i = f(x_i) + eps_i` with independent
  `eps_i ~ N(0, sigma_n^2)`.
- The latent function has a zero-mean GP prior
  `f ~ GP(0, k_theta(x, x'))`. (For non-zero-centered data, center `y`
  before training.)
- The kernel `k_theta` is stationary (RBF/Matern) or dot-product
  (linear), parameterized by strictly positive hyperparameters.

Training data `X in R^{n x d}`, `y in R^n` give

```
K_y = K_theta(X, X) + sigma_n^2 I
```

and the log marginal likelihood

```
log p(y | X, theta) = -1/2 y^T K_y^{-1} y
                      -1/2 log |K_y| - n/2 log(2 pi).
```

Hyperparameters are chosen by maximizing the LML.

Prediction at `x*`, using `alpha = K_y^{-1} y`:

```
mean*       = k_*^T alpha
var_latent* = k(x*,x*) - k_*^T K_y^{-1} k_*
var_obs*    = var_latent* + sigma_n^2
```

`gp.Prediction.Variance` is the latent posterior variance (it approaches
the noise level at observation points and tends to zero with negligible
noise); `ObservedVariance` predicts a fresh noisy measurement. Far from
observations the latent variance grows toward the kernel prior variance,
as required.

### Kernels

All stationary kernels support one scalar length scale or `d` ARD length
scales.

- RBF: `sigma_f^2 exp(-1/2 sum_d ((x_d-x'_d)/l_d)^2)`
- Matern-3/2: `sigma_f^2 (1 + sqrt(3) r) exp(-sqrt(3) r)`
- Matern-5/2: `sigma_f^2 (1 + sqrt(5) r + (5/3) r^2) exp(-sqrt(5) r)`
- Linear: `sigma_b^2 + sigma_v^2 x^T x'`

Parameters (natural, strictly positive):

- `l_1..l_d` length scales; `sigma_f^2` signal variance.
- Linear: `sigma_b^2`, `sigma_v^2`.
- `sigma_n^2` observation noise; optimized only when `OptimizeNoise`.

### Optimization

- Parameters are optimized in log space (`log l`, `log sigma_f^2`, and
  `log sigma_n^2`); log values are bounded to `[log 1e-10, log 1e6]`.
- L-BFGS (memory 10, two-loop recursion) with analytic LML gradients;
  step selection uses Armijo backtracking (`c1 = 1e-4`, shrink 0.5, up to
  40 trials). If a proposed direction is not descending the optimizer
  restarts with steepest descent.
- Termination reasons (`gp.TerminationReason`):
  `gradient_tol` (`max|grad| <= GradientTol`), `converged`
  (relative objective change `< ConvergenceTol`), `max_iterations`
  (`MaxIterations` accepted moves), or `line_search_failed`.
- LML monotonicity: if the search cannot improve the starting point, the
  initial parameters are restored, so `FinalLML >= InitialLML`.

## Numerical stability

`Options.Stability` selects the policy:

- `jitter` (default): on a failed Cholesky decomposition, add
  `InitialJitter` (default `1e-10`) to the diagonal and retry, multiplying
  by `JitterFactor` (default `10`) up to `MaxJitter` (default `1e-2`).
  The applied jitter is reported in `OptimizeProgress.AppliedJitter`.
- `strict`: never modify the matrix; return a wrapped
  `linalg.ErrNotSPD` error. Duplicate points with tiny noise therefore
  produce an identifiable error rather than a wrong answer.

Additional guards:

- Cholesky rejects non-positive pivots and NaN/Inf factors.
- Solved vectors are checked for NaN/Inf.
- Prediction variance is clamped to zero only for sub-`1e-8` round-off;
  significantly negative values return an error.
- Inputs are validated for dimension, NaN and Inf.

## Persistence and crash recovery

Checkpoint layout (`store.FileStore`):

- `checkpoint.json` – the single committed checkpoint;
- `checkpoint.json.tmp` – staged content, renamed into place only after a
  full write and `fsync`; the parent directory is fsynced so the rename is
  durable. Readers therefore see only a complete old or new checkpoint.

Each file is a JSON envelope `{checksum_algo, checksum, payload}` with a
SHA-256 over the exact payload bytes. On load the checksum is verified and
the payload is semantically validated (schema version, dimensions,
parameter validity, non-negative noise, consistent x/y). Any failure
returns `store.ErrCheckpointCorrupt` (or `store.ErrNoCheckpoint`) and the
model is not loaded.

Set `Options.CheckpointEvery = N` to persist after every N accepted
optimizer iterations. After a crash, `service.Restore(..., resume=true)`
loads the latest checkpoint, reconstructs the L-BFGS curvature history
from the recorded steps, and runs only new iterations: iteration numbers
continue from the stored count and finished iterations are never applied
twice.

## Concurrency and versions

- Each model keeps an `atomic.Pointer[VersionedModel]`. Updates
  (train/append/resume) for one model are serialized by a per-model mutex;
  queries load the pointer without that lock and run on the immutable
  snapshot.
- Models are published only after training completes and the factorization
  is precomputed (`Model.Prepare`), so prediction is read-only and cannot
  race with an update. Readers observe either the previous complete
  version or the new complete version, never mixed parameters.
- `go test -race ./...` is clean.

## Incremental append

`AppendObservations` merges new points into the stored design data, builds
a fresh kernel from the same initial kernel configuration and reruns the
same deterministic optimization on the full dataset. Consequently:

```
train(all)  ==  train(first) + append(rest)
```

up to floating-point differences. With identical inputs and options the
two paths execute the same arithmetic order, so in practice optimized
parameters and predictions match bit-for-bit; the documented guarantee is a
relative tolerance of `1e-6` for parameters and an absolute tolerance of
`1e-6` for predictive mean/variance (tested in
`service.TestAppendMatchesFullRetrain`).

## Determinism and floating-point rules

- Optimization is deterministic: it uses only arithmetic, no random
  numbers, and starts from the same initial parameters.
- Data generation (`dataset`) uses a private `math/rand.Rand` seeded by
  the caller (`RandomSeed`); the global rand source is never touched.
- Same inputs + same options + same seed => identical kernel parameters,
  noise and predictions across processes (tested in
  `service.TestDeterminismAcrossRuns`).
- FP comparison conventions used in tests:
  - analytic gradients vs central differences: absolute `1e-4` (kernel),
    relative `1e-3` on LML log-parameter gradients;
  - append-vs-retrain: relative `1e-6` parameters, absolute `1e-6`
    predictions;
  - tiny-noise interpolation: `|mean - y| < 1e-4`;
  - variance non-negativity is exact; near-noise observation variance uses
    an absolute tolerance of `1e-2` (finite-data posterior can be slightly
    below `sigma_n^2`).

## Complexity

For `n` observations, `d` dimensions:

- Covariance build: `O(n^2 d)`; Cholesky: `O(n^3)`.
- One LML/gradient evaluation: `O(n^3 + n^2 (d + p))` where `p` is the
  number of kernel parameters (`d+1` for ARD stationary kernels).
- One L-BFGS iteration performs several line-search evaluations; at most
  `MaxIterations` accepted moves are run.
- Prediction: mean `O(n^2)` per point (one cached solve path, dominated by
  triangular solves); batch predictions are `O(q n^2)` for `q` queries.
- Checkpoint storage: `O(n^2)` JSON size (full design matrices are stored
  for exact recovery).

## Repository layout

| Path | Responsibility |
| --- | --- |
| `kernel/` | Kernel types, parameters, covariance and analytic gradients |
| `internal/linalg/` | Cholesky factorization and SPD solves |
| `gp/` | Model, prediction, LML, L-BFGS optimizer, stability options |
| `store/` | Atomic, checksummed JSON checkpoint persistence |
| `service/` | Versioned concurrent service, append, restore |
| `dataset/` | Seeded local data generators |
| `cmd/gprs/` | Runnable `demo` and `serve` entry points |

## Tests

```sh
go test ./...          # all packages
go test -race ./...    # concurrency safety
```

Coverage includes: per-kernel analytic gradients, LML finite-difference
checks for every kernel, hyperparameter optimization and monotonicity,
ill-conditioned/duplicate inputs in both policies, crash recovery without
re-running iterations, corrupt/half-written checkpoint rejection,
incremental append equivalence, concurrent read/update, and determinism.
