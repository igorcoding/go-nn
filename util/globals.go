package util

type InputOutputType float64

type Row_t []float64
type Matrix_t []Row_t

func NewMatrix(rows, cols int) Matrix_t {
	m := make(Matrix_t, rows)
	for l := range(m) {
		m[l] = NewRow(cols)
	}
	return m
}

func NewRow(n int) Row_t {
	return make(Row_t, n)
}

type TrainExample struct {
	Input []float64
	Output []float64
}

func NewTrainExample(input, output []float64) TrainExample {
	return TrainExample{Input:input, Output:output}
}