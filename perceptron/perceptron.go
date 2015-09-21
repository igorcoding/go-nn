package perceptron
import (
	"fmt"
	"math/rand"
	"time"
	"github.com/igorcoding/go-nn/util"
)

import "C"

var _ = fmt.Printf

const (
	RAND_EPSILON = 0.6
)

type PerceptronConf struct {
	Inputs int
	LearningRate float64
	Iterations int
	Threshold float64
}

type perceptronNet struct {
	conf *PerceptronConf
	w []float64

	randomSource *rand.Rand
}

func BuildPerceptronNet(conf *PerceptronConf) *perceptronNet {
	if conf == nil {
		panic("Need config to build FFNet")
	}
	if conf.Inputs < 1 {
		panic("Number of inputs should be >= 1")
	}

	if conf.Iterations == 0 {
		conf.Iterations = 100
	}

	nn := &perceptronNet{conf:conf}
	nn.randomSource = rand.New(rand.NewSource(time.Now().UnixNano()))

	nn.w = make([]float64, conf.Inputs);
//	for i := range(nn.w) {
//		nn.w[i] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
//	}
	return nn
}

func (self *perceptronNet) Train(trainSet []util.TrainExample) {
	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
//		fmt.Println("Iteration #", iteration + 1)

		for k := range(trainSet) {
			input := trainSet[k].Input
			output := trainSet[k].Output
			y := output[0]

			h := self.forward(input)
			delta := self.conf.LearningRate * (y - h)


			for i := range(self.w) {
				self.w[i] += delta * input[i]
			}
		}
	}
}

func (self *perceptronNet) Predict(input []float64) []float64 {
	return []float64{self.forward(input)}
}

func (self *perceptronNet) forward(input []float64) float64 {
	if (len(input) != len(self.w)) {
		panic("Sizes don't match")
	}
	s := 0.0
	for i := range(self.w) {
		s += input[i] * self.w[i]
	}

	return self.threshold(s)
}

func (self *perceptronNet) threshold(value float64) float64 {
	if value <= self.conf.Threshold {
		return 0
	} else {
		return 1
	}
}

