package gp

const (
	defaultNoise          = 1e-2
	defaultMaxIterations  = 100
	defaultConvergenceTol = 1e-6
	defaultGradientTol    = 1e-6
	defaultInitialJitter  = 1e-10
	defaultMaxJitter      = 1e-2
	defaultJitterFactor   = 10.0
)

// withDefaults returns a copy of o with unset fields replaced by defaults.
func (o Options) withDefaults() Options {
	if o.NoiseVariance == 0 {
		o.NoiseVariance = defaultNoise
	}
	if o.MaxIterations == 0 {
		o.MaxIterations = defaultMaxIterations
	}
	if o.ConvergenceTol == 0 {
		o.ConvergenceTol = defaultConvergenceTol
	}
	if o.GradientTol == 0 {
		o.GradientTol = defaultGradientTol
	}
	if o.Stability == "" {
		o.Stability = StabilityJitter
	}
	if o.InitialJitter == 0 {
		o.InitialJitter = defaultInitialJitter
	}
	if o.MaxJitter == 0 {
		o.MaxJitter = defaultMaxJitter
	}
	if o.JitterFactor == 0 {
		o.JitterFactor = defaultJitterFactor
	}
	return o
}

// StabilityStrategy defines how the service handles ill-conditioned
// covariance matrices.
type StabilityStrategy string

const (
	// StabilityJitter automatically adds progressively larger diagonal
	// jitter until the Cholesky factorization succeeds.
	StabilityJitter StabilityStrategy = "jitter"
	// StabilityStrict never modifies the matrix and returns an error when
	// it is not positive definite.
	StabilityStrict StabilityStrategy = "strict"
)

// Options controls training, optimization and numerical stability.
type Options struct {
	// NoiseVariance is the observation noise variance (sigma_n^2).
	NoiseVariance float64
	// OptimizeNoise makes the noise variance a trainable hyperparameter.
	OptimizeNoise bool
	// MaxIterations bounds the optimizer (0 means a package default).
	MaxIterations int
	// ConvergenceTol stops optimization when relative LML improvement is
	// below this value.
	ConvergenceTol float64
	// GradientTol stops optimization when the gradient max norm is small.
	GradientTol float64
	// Stability selects the ill-conditioning policy.
	Stability StabilityStrategy
	// InitialJitter is the first diagonal perturbation tried.
	InitialJitter float64
	// MaxJitter bounds the jitter search.
	MaxJitter float64
	// JitterFactor grows the jitter between attempts.
	JitterFactor float64
	// RandomSeed seeds stochastic components (data generation only).
	RandomSeed int64
	// CheckpointEvery > 0 invokes CheckpointHook after every N accepted
	// optimizer iterations, enabling crash-safe incremental persistence.
	CheckpointEvery int
	// CheckpointHook receives an immutable snapshot of progress; callers
	// typically persist it atomically. It must not mutate the model.
	CheckpointHook func(progress *OptimizeProgress)
}

// DefaultOptions returns sane default training options.
func DefaultOptions() Options {
	return Options{}
}

// TerminationReason explains why optimization stopped.
type TerminationReason string

const (
	TerminatedConverged TerminationReason = "converged"
	TerminatedGradient  TerminationReason = "gradient_tol"
	TerminatedMaxIters  TerminationReason = "max_iterations"
	TerminatedFailed    TerminationReason = "line_search_failed"
)

// OptimizeStep records one optimizer iteration for checkpointing.
type OptimizeStep struct {
	Iteration  int       `json:"iteration"`
	Params     []float64 `json:"params"`
	NegLML     float64   `json:"neg_lml"`
	GradNorm   float64   `json:"grad_norm"`
	StepLength float64   `json:"step_length"`
}

// OptimizeProgress summarizes a hyperparameter optimization run.
type OptimizeProgress struct {
	InitialLML     float64           `json:"initial_lml"`
	FinalLML       float64           `json:"final_lml"`
	InitialParams  []float64         `json:"initial_params"`
	FinalParams    []float64         `json:"final_params"`
	Iterations     int               `json:"iterations"`
	Converged      bool              `json:"converged"`
	Termination    TerminationReason `json:"termination_reason"`
	AppliedJitter  float64           `json:"applied_jitter"`
	CompletedSteps []OptimizeStep    `json:"completed_steps"`
}
