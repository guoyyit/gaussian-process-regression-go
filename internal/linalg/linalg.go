package linalg

import (
	"errors"
	"math"
)

// ErrNotSPD indicates that a matrix that must be symmetric positive
// definite was not (numerically) positive definite.
var ErrNotSPD = errors.New("matrix is not symmetric positive definite")

const tiny = 1e-300

// Cholesky computes the lower-triangular Cholesky factor L of A such
// that A = L L^T.
func Cholesky(a [][]float64) ([][]float64, error) {
	n := len(a)
	l := make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := a[i][j]
			for k := 0; k < j; k++ {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 || math.IsNaN(sum) {
					return nil, ErrNotSPD
				}
				l[i][j] = math.Sqrt(sum)
			} else {
				v := sum / l[j][j]
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return nil, ErrNotSPD
				}
				l[i][j] = v
			}
		}
	}
	return l, nil
}

// SolveCholesky returns X solving A X = B, given the lower factor L of A.
// It uses one forward and one back substitution per right-hand side.
func SolveCholesky(l [][]float64, b [][]float64) [][]float64 {
	n := len(l)
	nrhs := len(b[0])
	x := make([][]float64, n)
	for i := range x {
		x[i] = make([]float64, nrhs)
	}
	for c := 0; c < nrhs; c++ {
		// Forward substitution: L Z = B.
		for i := 0; i < n; i++ {
			sum := b[i][c]
			for k := 0; k < i; k++ {
				sum -= l[i][k] * x[k][c]
			}
			x[i][c] = sum / l[i][i]
		}
		// Back substitution: L^T X = Z.
		for i := n - 1; i >= 0; i-- {
			sum := x[i][c]
			for k := i + 1; k < n; k++ {
				sum -= l[k][i] * x[k][c]
			}
			x[i][c] = sum / l[i][i]
		}
	}
	return x
}

// SolveSPD returns X solving A X = B for symmetric positive definite A.
func SolveSPD(a, b [][]float64) ([][]float64, error) {
	l, err := Cholesky(a)
	if err != nil {
		return nil, err
	}
	return SolveCholesky(l, b), nil
}

// DeterminantLog returns log(det(A)) given the lower Cholesky factor L of A.
func DeterminantLog(l [][]float64) float64 {
	sum := 0.0
	for i := range l {
		v := l[i][i]
		if v < tiny {
			v = tiny
		}
		sum += math.Log(v)
	}
	return 2 * sum
}

// DiagOfInverse returns the diagonal elements of A^{-1}.
func DiagOfInverse(l [][]float64) []float64 {
	n := len(l)
	id := make([][]float64, n)
	for i := range id {
		id[i] = make([]float64, n)
		id[i][i] = 1
	}
	inv := SolveCholesky(l, id)
	diag := make([]float64, n)
	for i := range diag {
		diag[i] = inv[i][i]
	}
	return diag
}

// CloneMatrix returns a deep copy of a.
func CloneMatrix(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = append([]float64(nil), a[i]...)
	}
	return out
}

// Dot returns the Euclidean inner product of a and b.
func Dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
