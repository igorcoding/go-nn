package util

type InputOutputType float64

type Row_t []float64
type Matrix_t []Row_t

type TrainExample struct {
	Input []float64
	Output []float64
}

func NewTrainExample(input, output []float64) TrainExample {
	return TrainExample{Input:input, Output:output}
}