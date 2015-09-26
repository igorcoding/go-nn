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
	w []float64

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

	nn.w = make([]float64, conf.Inputs);
//	for i := range(nn.w) {
//		nn.w[i] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
//	}
	return nn
}

func (self *PerceptronNet) Train(trainSet []util.TrainExample) error {
	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
//		fmt.Println("Iteration #", iteration + 1)

		for k := range(trainSet) {
			input := trainSet[k].Input
			output := trainSet[k].Output
			y := output[0]

			h, err := self.forward(input)
			if err != nil {
				return errors.New(fmt.Sprintf("traiSet[%d] input size problem", k))
			}
			delta := self.conf.LearningRate * (y - h)


			for i := range(self.w) {
				self.w[i] += delta * input[i]
			}
		}
	}
	return nil
}

func (self *PerceptronNet) Predict(input []float64) ([]float64, error) {
	value, err := self.forward(input)
	return []float64{value}, err
}

func (self *PerceptronNet) forward(input []float64) (float64, error) {
	if (len(input) != len(self.w)) {
		return 0.0, errors.New("Sizes don't match")
	}
	s := 0.0
	for i := range(self.w) {
		s += input[i] * self.w[i]
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

