package main
import (
	"fmt"
	"github.com/igorcoding/go-nn/util"
	"github.com/igorcoding/go-nn/ffnet"
	"github.com/igorcoding/go-nn/perceptron"
)


func main() {
	perceptron_test()
//	xor()
//	lecture()
}

func perceptron_test() {
	trainSet := []util.TrainExample{
		util.TrainExample{Input:[]float64{1, 1, 1,
			                              0, 0, 0,
		                                  0, 0, 0}, Output:[]float64{0}},
		util.TrainExample{Input:[]float64{0, 0, 0,
										  1, 1, 1,
										  0, 0, 0}, Output:[]float64{0}},
		util.TrainExample{Input:[]float64{0, 0, 0,
			                              0, 0, 0,
			                              1, 1, 1}, Output:[]float64{0}},

		util.TrainExample{Input:[]float64{1, 0, 0,
			                              1, 0, 0,
			                              1, 0, 0}, Output:[]float64{1}},
		util.TrainExample{Input:[]float64{0, 1, 0,
			                              0, 1, 0,
			                              0, 1, 0}, Output:[]float64{1}},
		util.TrainExample{Input:[]float64{0, 0, 1,
			                              0, 0, 1,
			                              0, 0, 1}, Output:[]float64{1}},
	}


	conf := &perceptron.PerceptronConf{
		Inputs: 9,
		LearningRate: 1,
		Iterations: 1000,
		Threshold: 20,
	}
	net := perceptron.BuildPerceptronNet(conf)
	net.Train(trainSet)

	for t := range(trainSet) {
		prediction, _ := net.Predict(trainSet[t].Input)
		fmt.Println(trainSet[t].Input, "=>", prediction)
	}
}

func xor() {
	trainSet := []util.TrainExample{
		util.TrainExample{Input:[]float64{0, 0}, Output:[]float64{0}},
		util.TrainExample{Input:[]float64{0, 1}, Output:[]float64{1}},
		util.TrainExample{Input:[]float64{1, 0}, Output:[]float64{1}},
		util.TrainExample{Input:[]float64{1, 1}, Output:[]float64{0}},
	}


	conf := &ffnet.FFNetConf{Layers: []int{2, 3, 1},
		LearningRate: 0.7,
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

func lecture() {
	trainSet := []util.TrainExample{
		util.NewTrainExample([]float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1}, []float64{1}),
		util.NewTrainExample([]float64{
			0, 0, 1,
			0, 1, 0,
			1, 0, 0}, []float64{1}),

		util.NewTrainExample([]float64{
			1, 1, 0,
			0, 1, 0,
			0, 0, 1}, []float64{0}),
		util.NewTrainExample([]float64{
			0, 0, 1,
			0, 1, 0,
			0, 0, 1}, []float64{0}),
		util.NewTrainExample([]float64{
			0, 1, 0,
			0, 1, 0,
			0, 1, 0}, []float64{0}),
		util.NewTrainExample([]float64{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1}, []float64{0}),
		util.NewTrainExample([]float64{
			1, 1, 0,
			0, 1, 0,
			0, 1, 1}, []float64{0}),
		util.NewTrainExample([]float64{
			1, 0, 1,
			0, 1, 0,
			1, 0, 1}, []float64{0}),
	}

	conf := &ffnet.FFNetConf{Layers: []int{9, 3, 1},
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