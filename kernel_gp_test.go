package gpr

import (
	"math"
	"testing"
)

func approx(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestKernelValues(t *testing.T) {
	p := KernelParams{LengthScale: 2, SignalVariance: 3, Alpha: 2}
	a, b := []float64{0, 0}, []float64{2, 0}
	r2 := 4.0

	rbf, err := NewKernel(KernelRBF, p)
	if err != nil {
		t.Fatal(err)
	}
	wantRBF := 3 * math.Exp(-r2/(2*4))
	if !approx(rbf.Value(a, b), wantRBF, 1e-12) {
		t.Fatalf("rbf off-diag = %v want %v", rbf.Value(a, b), wantRBF)
	}
	if rbf.Value(a, a) != 3 {
		t.Fatalf("rbf diagonal = %v want 3", rbf.Value(a, a))
	}

	mat, err := NewKernel(KernelMatern, p)
	if err != nil {
		t.Fatal(err)
	}
	z := math.Sqrt(3) * 2 / 2
	wantMat := 3 * (1 + z) * math.Exp(-z)
	if !approx(mat.Value(a, b), wantMat, 1e-12) {
		t.Fatalf("matern off-diag = %v want %v", mat.Value(a, b), wantMat)
	}

	lin, err := NewKernel(KernelLinear, p)
	if err != nil {
		t.Fatal(err)
	}
	if !approx(lin.Value([]float64{1, 2}, []float64{3, 4}), 3*11.0/4.0, 1e-12) {
		t.Fatal("linear kernel value wrong")
	}

	rq, err := NewKernel(KernelRQ, p)
	if err != nil {
		t.Fatal(err)
	}
	wantRQ := 3 * math.Pow(1+r2/(2*2*4), -2)
	if !approx(rq.Value(a, b), wantRQ, 1e-12) {
		t.Fatalf("rq off-diag = %v want %v", rq.Value(a, b), wantRQ)
	}

	if _, err := NewKernel(KernelType("nope"), p); err != ErrUnknownKernel {
		t.Fatalf("want ErrUnknownKernel, got %v", err)
	}
	bad := p
	bad.LengthScale = -1
	if _, err := NewKernel(KernelRBF, bad); err == nil {
		t.Fatal("expected error for negative length scale")
	}
}

func TestDatasetValidation(t *testing.T) {
	if _, err := NewDataset(nil, nil); err != ErrEmptyData {
		t.Fatalf("want ErrEmptyData, got %v", err)
	}
	if _, err := NewDataset([][]float64{{1}, {2, 3}}, []float64{1, 2}); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
	if _, err := NewDataset([][]float64{{1}, {2}}, []float64{1}); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch for y, got %v", err)
	}
	if _, err := NewDataset([][]float64{{1}, {math.NaN()}}, []float64{1, 2}); err == nil {
		t.Fatal("expected NaN rejection")
	}
	d, err := NewDataset([][]float64{{1}, {2}}, []float64{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	d2, err := d.Append([][]float64{{3}}, []float64{5})
	if err != nil {
		t.Fatal(err)
	}
	if d2.Len() != 3 || d.Len() != 2 {
		t.Fatal("append must not mutate original dataset")
	}
}

// gpFromData 用指定核与噪声拟合模型。
func gpFromData(t *testing.T, x [][]float64, y []float64, kt KernelType, p KernelParams, noise float64) *Model {
	t.Helper()
	data, err := NewDataset(x, y)
	if err != nil {
		t.Fatal(err)
	}
	k, err := NewKernel(kt, p)
	if err != nil {
		t.Fatal(err)
	}
	model, err := fitGP(data, k, noise, meanValue(y), DefaultStabilityConfig())
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestGPRegressionProperties(t *testing.T) {
	xs := [][]float64{{-1}, {0}, {1}, {2}}
	ys := []float64{-1, 0, 1, 2}

	for _, kt := range []KernelType{KernelRBF, KernelMatern, KernelRQ} {
		t.Run(string(kt), func(t *testing.T) {
			model := gpFromData(t, xs, ys, kt, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}, 1e-6)

			// 噪声极小时，观测点处均值逼近观测值，方差接近噪声方差。
			for i, x := range xs {
				p, err := model.Predict(x)
				if err != nil {
					t.Fatal(err)
				}
				if math.Abs(p.Mean-ys[i]) > 1e-4 {
					t.Fatalf("observed mean = %v want %v", p.Mean, ys[i])
				}
				if p.Variance < 0 {
					t.Fatalf("negative variance %v", p.Variance)
				}
				if p.Variance > 1e-3 {
					t.Fatalf("variance at observed point %v too large", p.Variance)
				}
			}

			// 远离观测点方差单调趋大。
			pNear, _ := model.Predict([]float64{0})
			pFar, err := model.Predict([]float64{10})
			if err != nil {
				t.Fatal(err)
			}
			if !(pFar.Variance > pNear.Variance) {
				t.Fatalf("far variance %v should exceed near variance %v", pFar.Variance, pNear.Variance)
			}
		})
	}
}

func TestGPMultidimensional(t *testing.T) {
	xs := [][]float64{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
	}
	ys := []float64{0, 1, 1, 2}
	model := gpFromData(t, xs, ys, KernelRBF, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}, 1e-6)

	out, err := model.PredictBatch([][]float64{{0.5, 0.5}, {5, 5}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatal("batch size mismatch")
	}
	for _, p := range out {
		if !(p.Variance >= 0) || math.IsNaN(p.Mean) || math.IsInf(p.Variance, 0) {
			t.Fatalf("bad prediction %+v", p)
		}
	}
	if _, err := model.Predict([]float64{1}); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

func TestObservedPointVarianceIsNoiseLevel(t *testing.T) {
	xs := [][]float64{{-1}, {0}, {1}, {2}}
	ys := []float64{-1, 0, 1, 2}
	noise := 0.05
	model := gpFromData(t, xs, ys, KernelRBF,
		KernelParams{LengthScale: 1, SignalVariance: 2, Alpha: 1}, noise)
	for i, x := range xs {
		p, err := model.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		// 后验（条件）方差在观测点处趋近噪声方差 σ²，而非 0。
		if !approx(p.Variance, noise*noise, 1e-4) {
			t.Fatalf("point %d variance = %v want ~%v", i, p.Variance, noise*noise)
		}
	}
	// 远离观测点方差趋近先验 k(x,x)=signal variance。
	pFar, err := model.Predict([]float64{50})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(pFar.Variance-2.0) > 1e-3 {
		t.Fatalf("far variance = %v want prior ~2", pFar.Variance)
	}
}

func TestTinyNoiseInterpolatesObservations(t *testing.T) {
	// 噪声取极小时，GP 应在观测点近乎完美插值。
	xs := [][]float64{{-2}, {-0.5}, {0.5}, {2}}
	ys := []float64{0.3, -1.2, 0.8, 2.4}
	model := gpFromData(t, xs, ys, KernelRBF,
		KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}, 1e-7)
	for i, x := range xs {
		p, err := model.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(p.Mean-ys[i]) > 1e-5 {
			t.Fatalf("tiny-noise mean = %v want %v", p.Mean, ys[i])
		}
	}

	// 显式关闭抖动且数据良态时也必须正常工作。
	data, _ := NewDataset(xs, ys)
	k, _ := NewKernel(KernelRBF, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1})
	off, err := fitGP(data, k, 1e-3, meanValue(ys), StabilityConfig{Policy: JitterOff})
	if err != nil {
		t.Fatalf("well-conditioned fit with JitterOff failed: %v", err)
	}
	p, err := off.Predict([]float64{0.5})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || p.Variance < 0 {
		t.Fatalf("bad prediction with jitter off: %+v", p)
	}
}

func TestLinearKernelMultidimensional(t *testing.T) {
	// 线性核 K=XXᵀ/ℓ² 在 n>d 时秩亏，天然病态：
	// JitterOff 必须报可识别错误；自适应抖动下结果必须稳定有限。
	xs := [][]float64{{1, 1}, {2, 1}, {1, 2}, {3, 1}}
	ys := []float64{3, 4, 5, 5}
	data, _ := NewDataset(xs, ys)
	k, _ := NewKernel(KernelLinear, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1})
	if _, err := fitGP(data, k, 1e-6, meanValue(ys),
		StabilityConfig{Policy: JitterOff}); err != ErrSingular {
		t.Fatalf("rank-deficient linear kernel: want ErrSingular, got %v", err)
	}
	model := gpFromData(t, xs, ys, KernelLinear,
		KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1}, 1e-6)
	p, err := model.Predict([]float64{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	// 零均值线性核在训练点处精确插值；域外预测向零（中心化后为样本均值）回归。
	if math.IsNaN(p.Mean) || math.IsInf(p.Mean, 0) {
		t.Fatalf("non-finite linear mean: %v", p.Mean)
	}
	if p.Variance < 0 {
		t.Fatalf("negative variance %v", p.Variance)
	}
}

func TestIllConditionedDuplicates(t *testing.T) {
	// 完全重合的输入点 + 零噪声使 K 奇异；自适应抖动应保证可用结果。
	xs := [][]float64{{1}, {1}, {2}}
	ys := []float64{3, 3, 4}
	data, _ := NewDataset(xs, ys)
	k, _ := NewKernel(KernelRBF, KernelParams{LengthScale: 1, SignalVariance: 1, Alpha: 1})

	// 关闭抖动时必须返回可识别错误，而不是 NaN/Inf。
	if _, err := fitGP(data, k, 0, meanValue(ys), StabilityConfig{Policy: JitterOff}); err != ErrSingular {
		t.Fatalf("want ErrSingular with JitterOff, got %v", err)
	}

	// 自适应抖动下拟合成功，结果有限。
	model, err := fitGP(data, k, 0, meanValue(ys), DefaultStabilityConfig())
	if err != nil {
		t.Fatalf("adaptive jitter should recover, got %v", err)
	}
	p, err := model.Predict([]float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p.Mean) || math.IsInf(p.Mean, 0) || math.IsNaN(p.Variance) || math.IsInf(p.Variance, 0) {
		t.Fatalf("non-finite prediction with duplicates: %+v", p)
	}
}
