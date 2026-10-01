package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"sync"

	gpr "github.com/guoyyit/gaussian-process-regression-go"
)

func main() {
	dir := flag.String("dir", ".gpr-data", "checkpoint directory")
	kernelName := flag.String("kernel", "rbf", "kernel: rbf|matern32|linear|rational_quadratic")
	iters := flag.Int("iters", 100, "maximum optimizer iterations")
	points := flag.Int("points", 40, "number of synthetic observations")
	seed := flag.Int64("seed", 42, "fixed RNG seed for data generation")
	flag.Parse()

	xs, ys := synthSin(*points, *seed)

	svc, err := gpr.NewService(gpr.ServiceConfig{CheckpointDir: *dir, Seed: *seed})
	if err != nil {
		fail(err)
	}
	defer svc.Close()

	// 若已有完整检查点则恢复，否则训练。
	if err := svc.Restore(); err != nil {
		fmt.Printf("no restorable checkpoint (%v); training from scratch\n", err)
		cfg := gpr.TrainConfig{
			KernelType:    gpr.KernelType(*kernelName),
			InitialParams: gpr.KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1},
			Noise:         0.1,
			Optimize:      true,
			Stability:     gpr.DefaultStabilityConfig(),
		}
		cfg.Optimizer = gpr.OptimizerConfig{
			MaxIterations: *iters, Tolerance: 1e-6, LearningRate: 0.05,
			GradientEpsilon: 1e-5, NoImproveWindow: 15, Seed: *seed,
			CheckpointEvery: 10,
		}
		res, err := svc.Train(context.Background(), xs, ys, cfg)
		if err != nil {
			fail(err)
		}
		printResult("train", res)
	} else {
		info, _ := svc.CurrentVersion()
		fmt.Printf("restored model: iterations=%d completed=%v finalLML=%.4f\n",
			info.Iteration, info.Completed, info.FinalLML)
	}

	// 更新期间并发查询，验证读取不阻塞且不出现撕裂读。
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					if _, err := svc.Predict([]float64{0.42}); err != nil {
						fmt.Fprintln(os.Stderr, "predict:", err)
						return
					}
				}
			}
		}()
	}

	// 增量追加两个观测点。
	extraX := [][]float64{{3.05}, {3.1}}
	extraY := []float64{math.Sin(3.05), math.Sin(3.1)}
	res, err := svc.Append(context.Background(), extraX, extraY)
	if err != nil {
		fail(err)
	}
	close(stop)
	wg.Wait()
	printResult("append", res)

	for _, x := range []float64{-2.5, 0, 1.5, 3.1} {
		p, err := svc.Predict([]float64{x})
		if err != nil {
			fail(err)
		}
		fmt.Printf("predict(%5.2f): mean=%+.5f variance=%.6f (target %+.5f)\n",
			x, p.Mean, p.Variance, math.Sin(x))
	}
}

// synthSin 用确定性 LCG 生成带噪声的 sin 观测；固定种子即可复现。
func synthSin(n int, seed int64) ([][]float64, []float64) {
	state := uint64(seed)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state%10000)/10000.0 - 0.5
	}
	xs := make([][]float64, n)
	ys := make([]float64, n)
	for i := 0; i < n; i++ {
		x := -3 + 6*float64(i)/float64(n-1)
		xs[i] = []float64{x}
		ys[i] = math.Sin(x) + 0.02*next()
	}
	return xs, ys
}

func printResult(stage string, res *gpr.OptimizeResult) {
	fmt.Printf("[%s] iterations=%d stop=%s LML %.4f -> %.4f params=%+v noise=%.6f\n",
		stage, res.Iterations, res.StopReason, res.InitialLML, res.FinalLML,
		res.Kernel.Params(), res.Noise)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
