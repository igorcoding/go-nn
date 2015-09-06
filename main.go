package main
import (
	"fmt"
	"github.com/igorcoding/nn/ffnet"
)


func main() {

//	trainSet := []ffnet.TrainExample{
//		ffnet.NewTrainExample([]float64{
//			1, 0, 0,
//			0, 1, 0,
//			0, 0, 1}, []float64{1}),
//		ffnet.NewTrainExample([]float64{
//			0, 0, 1,
//			0, 1, 0,
//			1, 0, 0}, []float64{1}),
//
//		ffnet.NewTrainExample([]float64{
//			1, 1, 0,
//			0, 1, 0,
//			0, 0, 1}, []float64{0}),
//		ffnet.NewTrainExample([]float64{
//			0, 0, 1,
//			0, 1, 0,
//			0, 0, 1}, []float64{0}),
//		ffnet.NewTrainExample([]float64{
//			0, 1, 0,
//			0, 1, 0,
//			0, 1, 0}, []float64{0}),
//		ffnet.NewTrainExample([]float64{
//			1, 1, 1,
//			1, 1, 1,
//			1, 1, 1}, []float64{0}),
//		ffnet.NewTrainExample([]float64{
//			1, 1, 0,
//			0, 1, 0,
//			0, 1, 1}, []float64{0}),
//		ffnet.NewTrainExample([]float64{
//			1, 0, 1,
//			0, 1, 0,
//			1, 0, 1}, []float64{0}),
//	}

	trainSet := []ffnet.TrainExample{
		ffnet.TrainExample{Input:[]float64{0, 0}, Output:[]float64{0}},
		ffnet.TrainExample{Input:[]float64{0, 1}, Output:[]float64{1}},
		ffnet.TrainExample{Input:[]float64{1, 0}, Output:[]float64{1}},
		ffnet.TrainExample{Input:[]float64{1, 1}, Output:[]float64{0}},
	}



	conf := &ffnet.FFNetConf{Layers: []int32{2,2,1},
	                         Alpha: 0.8,
	                         Regularization: 0.8,
	                         Bias: true,
	                         Iterations: 10000}
	net := ffnet.BuildFFNet(conf)
	net.Train(trainSet)

	for t := range(trainSet) {
		fmt.Println(trainSet[t].Input, "=>", net.Predict(trainSet[t].Input))
	}
}