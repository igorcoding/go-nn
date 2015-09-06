package main
import (
	"fmt"
	"github.com/igorcoding/nn/ffnet"
)


func main() {
	xor()
//	lecture()
}

func xor() {
	trainSet := []ffnet.TrainExample{
		ffnet.TrainExample{Input:[]float64{0, 0}, Output:[]float64{0}},
		ffnet.TrainExample{Input:[]float64{0, 1}, Output:[]float64{1}},
		ffnet.TrainExample{Input:[]float64{1, 0}, Output:[]float64{1}},
		ffnet.TrainExample{Input:[]float64{1, 1}, Output:[]float64{0}},
	}


	conf := &ffnet.FFNetConf{Layers: []int32{2, 3, 1},
		LearningRate: 0.7,
		Momentum: 0.8,
		Regularization: 0.0001,
		Bias: false,
		Iterations: 10000}
	net := ffnet.BuildFFNet(conf)
	net.Train(trainSet)

	for t := range(trainSet) {
		fmt.Println(trainSet[t].Input, "=>", net.Predict(trainSet[t].Input))
	}
}

func lecture() {
	trainSet := []ffnet.TrainExample{
		ffnet.NewTrainExample([]float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1}, []float64{1}),
		ffnet.NewTrainExample([]float64{
			0, 0, 1,
			0, 1, 0,
			1, 0, 0}, []float64{1}),

		ffnet.NewTrainExample([]float64{
			1, 1, 0,
			0, 1, 0,
			0, 0, 1}, []float64{0}),
		ffnet.NewTrainExample([]float64{
			0, 0, 1,
			0, 1, 0,
			0, 0, 1}, []float64{0}),
		ffnet.NewTrainExample([]float64{
			0, 1, 0,
			0, 1, 0,
			0, 1, 0}, []float64{0}),
		ffnet.NewTrainExample([]float64{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1}, []float64{0}),
		ffnet.NewTrainExample([]float64{
			1, 1, 0,
			0, 1, 0,
			0, 1, 1}, []float64{0}),
		ffnet.NewTrainExample([]float64{
			1, 0, 1,
			0, 1, 0,
			1, 0, 1}, []float64{0}),
	}

	conf := &ffnet.FFNetConf{Layers: []int32{9, 3, 1},
		LearningRate: 0.2,
		Momentum: 0.8,
		Regularization: 0.0001,
		Bias: true,
		Iterations: 10000}
	net := ffnet.BuildFFNet(conf)
	net.Train(trainSet)

	for t := range(trainSet) {
		fmt.Println(trainSet[t].Input, "=>", net.Predict(trainSet[t].Input))
	}
}