package util
import "fmt"

type matrix struct {
	rows, cols int
	m [][]float64
}

type MatrixOP interface {
	Add(*matrix) *matrix
	AddNum(float64) *matrix
	Minus(*matrix) *matrix
	MinusNum(float64) *matrix
	MultiplyNum(float64) *matrix
	T() *matrix

	Rows() int
	Cols() int
	At(i, j int) float64
	Set(j, i int, v float64)
}


func NewMatrix(rows, cols int) *matrix {
	mat := &matrix{rows: rows, cols: cols}
	mat.m = make([][]float64, rows)
	for i := range(mat.m) {
		mat.m[i] = make([]float64, cols)
	}
	return mat
}

func (self *matrix) Rows() int {
	return self.rows
}

func (self *matrix) Cols() int {
	return self.cols
}

func (self *matrix) At(i, j int) float64 {
	return self.m[i][j]
}

func (self *matrix) Set(i, j int, v float64) {
	self.m[i][j] = v
}

func (self *matrix) Add(rhs *matrix) *matrix {
	if rhs == nil { return self }
	if self.rows != rhs.rows && self.cols != rhs.cols {
		panic(fmt.Sprintf("Matricies should have equal size. Got: %dx%d and %dx%d", self.rows, self.cols, rhs.rows, rhs.cols))
	}

	for i := range(self.m) {
		for j := range(self.m[i]) {
			self.m[i][j] += rhs.m[i][j]
		}
	}
	return self
}

func (self *matrix) AddNum(rhs float64) *matrix {
	for i := range(self.m) {
		for j := range(self.m[i]) {
			self.m[i][j] += rhs
		}
	}
	return self
}

func (self *matrix) Minus(rhs *matrix) *matrix {
	if rhs == nil { return self }
	if self.rows != rhs.rows && self.cols != rhs.cols {
		panic(fmt.Sprintf("Matricies should have equal size. Got: %dx%d and %dx%d", self.rows, self.cols, rhs.rows, rhs.cols))
	}

	for i := range(self.m) {
		for j := range(self.m[i]) {
			self.m[i][j] -= rhs.m[i][j]
		}
	}
	return self
}

func (self *matrix) MinusNum(rhs float64) *matrix {
	for i := range(self.m) {
		for j := range(self.m[i]) {
			self.m[i][j] -= rhs
		}
	}
	return self
}

func (self *matrix) MultiplyNum(rhs float64) *matrix {
	for i := range(self.m) {
		for j := range(self.m[i]) {
			self.m[i][j] *= rhs
		}
	}
	return self
}

func (self *matrix) T() *matrix {
	m := [][]float64(nil)
	if self.rows == self.cols {
		m = self.m
	} else {
		m = make([][]float64, self.cols)
		for i := range(m) {
			m[i] = make([]float64, self.rows)
		}
	}
	for i := 0; i < self.rows; i++ {
		for j := i; j < self.cols; j++ {
			temp := self.m[j][i]
			m[j][i] = self.m[i][j]
			m[i][j] = temp
		}
	}
	self.m = m
	return self
}

func AddMatricies(m1, m2 *matrix) *matrix {
	if m1 == nil { return m2 }
	if m2 == nil { return m1 }
	mat := &matrix{rows: m1.rows, cols: m1.cols}
	mat.m = make([][]float64, mat.rows)
	for i := range(mat.m) {
		mat.m[i] = make([]float64, mat.cols)
		for j := range(mat.m[i]) {
			mat.m[i][j] = m1.m[i][j] + m2.m[i][j]
		}
	}
	return mat
}

func SubtractMatricies(m1, m2 *matrix) *matrix {
	if m1.rows != m2.rows && m1.cols != m2.cols {
		panic(fmt.Sprintf("Matricies should have equal size. Got: %dx%d and %dx%d", m1.rows, m1.cols, m2.rows, m2.cols))
	}
	mat := &matrix{rows: m1.rows, cols: m1.cols}
	mat.m = make([][]float64, mat.rows)
	for i := range(mat.m) {
		mat.m[i] = make([]float64, mat.cols)
		for j := range(mat.m[i]) {
			mat.m[i][j] = m1.m[i][j] - m2.m[i][j]
		}
	}
	return mat
}

func Multiply(m1, m2 *matrix) *matrix {
	if m1 == nil || m2 == nil {
		panic("Matrices should not be nil")
	}
	if m1.cols != m2.rows {
		panic(fmt.Sprintf("Matricies have wrong sizes. Got: %dx%d and %dx%d", m1.rows, m1.cols, m2.rows, m2.cols))
	}

	mat := NewMatrix(m1.rows, m2.cols)
	for i1 := range(m1.m) {
		for j2 := range(m2.m) {
			for j1 := range(m1.m[i1]) {
				mat.m[i1][j2] += m1.m[i1][j1] * m2.m[j1][j2]
			}
		}
	}

	return mat
}