package dataset

import "testing"

func TestDeterministicAcrossGenerators(t *testing.T) {
	g1 := New(42, 1, Sinc1D)
	g2 := New(42, 1, Sinc1D)
	x1, y1 := g1.Sample(20, 0.05)
	x2, y2 := g2.Sample(20, 0.05)
	for i := range x1 {
		if x1[i][0] != x2[i][0] || y1[i] != y2[i] {
			t.Fatalf("non-deterministic sample at %d: %v/%v vs %v/%v", i, x1[i], y1[i], x2[i], y2[i])
		}
	}
	g3 := New(43, 1, Sinc1D)
	x3, _ := g3.Sample(20, 0.05)
	if x3[0][0] == x1[0][0] {
		t.Fatal("different seeds unexpectedly produced identical data")
	}
}

func TestGridEndpointsAndSpacing(t *testing.T) {
	g := Grid1D(-2, 2, 5)
	if g[0][0] != -2 || g[4][0] != 2 || len(g) != 5 {
		t.Fatalf("bad grid: %v", g)
	}
}

func TestSincAtZero(t *testing.T) {
	if v := Sinc1D([]float64{0}); v != 1 {
		t.Fatalf("sinc(0) = %v, want 1", v)
	}
}
