package main
import (
	"fmt"
	"github.com/igorcoding/nn/ffnet"
)


func main() {

	trainSet := []ffnet.TrainExample{
		ffnet.TrainExample{Input:[]float64{0, 0}, Output:[]float64{0}},
		ffnet.TrainExample{Input:[]float64{0, 1}, Output:[]float64{0}},
		ffnet.TrainExample{Input:[]float64{1, 0}, Output:[]float64{0}},
		ffnet.TrainExample{Input:[]float64{1, 1}, Output:[]float64{0}},
	}

	conf := &ffnet.FFNetConf{Layers: []int32{2,3,1},
	                         Alpha: 0.9,
							 Eta: 0.1,
							 Regularization: 0.8,
//							 Bias: true,
							 Iterations: 10}
	net := ffnet.BuildFFNet(conf)
	net.Train(trainSet)

	fmt.Println("{0, 0} => ", net.Predict([]float64{0, 0}))
	fmt.Println("{0, 1} => ", net.Predict([]float64{0, 1}))
	fmt.Println("{1, 0} => ", net.Predict([]float64{1, 0}))
	fmt.Println("{1, 1} => ", net.Predict([]float64{1, 1}))

	fmt.Println(net)
}