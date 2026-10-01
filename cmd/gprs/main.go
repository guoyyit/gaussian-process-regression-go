package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"

	"github.com/guoyyit/gaussian-process-regression-go/dataset"
	"github.com/guoyyit/gaussian-process-regression-go/gp"
	"github.com/guoyyit/gaussian-process-regression-go/kernel"
	"github.com/guoyyit/gaussian-process-regression-go/service"
	"github.com/guoyyit/gaussian-process-regression-go/store"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "demo":
		runDemo(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "gprs - local gaussian process regression service")
	fmt.Fprintln(os.Stderr, "usage: gprs <demo|serve> [flags]")
}

func runDemo(args []string) {
	fs := flag.NewFlagSet("demo", flag.ExitOnError)
	dir := fs.String("dir", ".gprs-demo", "checkpoint directory")
	seed := fs.Int64("seed", 42, "random seed for data generation")
	n := fs.Int("n", 40, "number of training points")
	noise := fs.Float64("noise", 0.02, "observation noise std")
	kernelName := fs.String("kernel", "rbf", "kernel: rbf|matern32|matern52")
	maxIter := fs.Int("iters", 60, "optimizer iteration cap")
	_ = fs.Parse(args)

	x, y := dataset.NoisyGrid(*seed, -5, 5, *n, *noise, dataset.Sinc1D)

	st, err := store.NewFileStore(*dir)
	must(err)
	svc := service.New(st)

	opts := gp.DefaultOptions()
	opts.NoiseVariance = (*noise) * (*noise)
	opts.OptimizeNoise = true
	opts.MaxIterations = *maxIter
	opts.RandomSeed = *seed

	kcfg := kernel.Config{Type: kernel.Type(*kernelName), InputDim: 1, LengthScales: []float64{2.0}}
	v, err := svc.Train(context.Background(), service.TrainRequest{
		ModelID: "demo",
		Kernel:  kcfg,
		X:       x,
		Y:       y,
		Options: opts,
	})
	must(err)

	prog := v.Model.Progress()
	fmt.Printf("trained v%d: kernel=%s iters=%d termination=%s converged=%t\n",
		v.Version, *kernelName, prog.Iterations, prog.Termination, prog.Converged)
	fmt.Printf("LML: initial=%.6f final=%.6f (monotone=%t)\n",
		prog.InitialLML, prog.FinalLML, prog.FinalLML >= prog.InitialLML-1e-9)
	fmt.Printf("params: %v noise=%.6g jitter=%.3g\n", v.Model.Kernel.Params(), v.Model.Noise, prog.AppliedJitter)

	for _, q := range []float64{-4, -2, 0, 2, 4, 6} {
		p, err := svc.Predict("demo", []float64{q})
		must(err)
		truth := 1.0
		if math.Abs(q) > 1e-12 {
			truth = math.Sin(q) / q
		}
		fmt.Printf("predict x=%5.1f mean=%8.4f std=%7.4f (truth=%.4f)\n",
			q, p.Mean, math.Sqrt(p.Variance), truth)
	}

	// Demonstrate restart + restore without rerunning iterations.
	svc2 := service.New(st)
	restored, err := svc2.Restore(context.Background(), "demo", kcfg, false)
	must(err)
	rp, _ := svc2.Predict("demo", []float64{0})
	fmt.Printf("restored v%d at x=0: mean=%.6f var=%.6f\n",
		restored.Version, rp.Mean, rp.Variance)
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "listen address")
	dir := fs.String("dir", ".gprs-data", "checkpoint directory")
	modelID := fs.String("model", "default", "model id to serve (must be trained/restored first)")
	ktype := fs.String("kernel", "rbf", "kernel for restore")
	dim := fs.Int("dim", 1, "input dimension for restore")
	_ = fs.Parse(args)

	st, err := store.NewFileStore(*dir)
	must(err)
	svc := service.New(st)
	if _, err := svc.Restore(context.Background(), *modelID,
		kernel.Config{Type: kernel.Type(*ktype), InputDim: *dim}, false); err != nil {
		fmt.Fprintf(os.Stderr, "warning: restore failed (%v); train a model first\n", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			X []float64 `json:"x"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		p, err := svc.Predict(*modelID, req.X)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]float64{
			"mean": p.Mean, "variance": p.Variance, "observed_variance": p.ObservedVariance,
		})
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})
	fmt.Printf("gprs listening on %s (model %q)\n", *addr, *modelID)
	must(http.ListenAndServe(*addr, mux))
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
