package perceptron
import (
	"fmt"
	"math/rand"
	"time"
	"github.com/igorcoding/go-nn/util"
	"errors"
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

type PerceptronNet struct {
	conf *PerceptronConf
	W []float64

	randomSource *rand.Rand
}

func BuildPerceptronNet(conf *PerceptronConf) *PerceptronNet {
	if conf == nil {
		panic("Need config to build FFNet")
	}
	if conf.Inputs < 1 {
		panic("Number of inputs should be >= 1")
	}

	if conf.Iterations == 0 {
		conf.Iterations = 100
	}

	nn := &PerceptronNet{conf:conf}
	nn.randomSource = rand.New(rand.NewSource(time.Now().UnixNano()))

	nn.W = make([]float64, conf.Inputs);
	for i := range(nn.W) {
		nn.W[i] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
	}
	return nn
}

type iteration_t map[string]interface{}

func (self *PerceptronNet) Train(trainSet []util.TrainExample) ([][]float64, []float64, error) {

	weights := make([][]float64, len(self.W))
	for i := range(weights) {
		weights[i] = make([]float64, self.conf.Iterations)
	}

	trainErrors := make([]float64, self.conf.Iterations)

	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
//		fmt.Println("Iteration #", iteration + 1)

		trainErrors[iteration] = 0

		for k := range(trainSet) {
			input := trainSet[k].Input
			output := trainSet[k].Output
			y := output[0]

			h, err := self.forward(input)
			if err != nil {
				return nil, nil, errors.New(fmt.Sprintf("traiSet[%d] input size problem", k))
			}
			d := y - h
			delta := self.conf.LearningRate * d
			trainErrors[iteration] += d * d


			for i := range(self.W) {
				self.W[i] += delta * input[i]
			}
		}

		trainErrors[iteration] /= 2 * float64(len(trainSet))

		for i := range(self.W) {
			weights[i][iteration] = self.W[i]
		}
	}
	return weights, trainErrors, nil
}

func (self *PerceptronNet) Predict(input []float64) ([]float64, error) {
	value, err := self.forward(input)
	return []float64{value}, err
}

func (self *PerceptronNet) forward(input []float64) (float64, error) {
	if (len(input) != len(self.W)) {
		return 0.0, errors.New("Sizes don't match")
	}
	s := 0.0
	for i := range(self.W) {
		s += input[i] * self.W[i]
	}

	return self.threshold(s), nil
}

func (self *PerceptronNet) threshold(value float64) float64 {
	if value <= self.conf.Threshold {
		return 0
	} else {
		return 1
	}
}

