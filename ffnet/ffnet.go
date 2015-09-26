package ffnet
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

type FFNetConf struct {
	Layers []int
	LearningRate float64
	Momentum float64
	Regularization float64
	Bias bool
	Iterations int
	Activation ActivationFunc
}

type FFNet struct {
	conf *FFNetConf
	w []util.Matrix_t
	b util.Matrix_t

	randomSource *rand.Rand
}

func BuildFFNet(conf *FFNetConf) *FFNet {
	if conf == nil {
		panic("Need config to build FFNet")
	}
	if conf.Layers == nil || len(conf.Layers) < 2 {
		panic("Number of layers should be >= 2")
	}
	if conf.Activation == nil {
		conf.Activation = Sigma
	}
	if conf.Iterations == 0 {
		conf.Iterations = 100
	}

	nn := &FFNet{conf:conf}
	nn.randomSource = rand.New(rand.NewSource(time.Now().UnixNano()))

	nn.w = make([]util.Matrix_t, len(conf.Layers) - 1);
	for l := range(nn.w) {
		nn.w[l] = util.NewMatrix(conf.Layers[l], conf.Layers[l + 1])
		for i := range(nn.w[l]) {
			for j := range(nn.w[l][i]) {
				nn.w[l][i][j] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
			}
		}
	}
	nn.b = make(util.Matrix_t, len(conf.Layers) - 1)
	for l := range(nn.b) {
		nn.b[l] = make(util.Row_t, conf.Layers[l+1])
		if conf.Bias {
			for j := range (nn.b[l]) {
				nn.b[l][j] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
			}
		}
	}
	return nn
}

func (self *FFNet) Train(trainSet []util.TrainExample) {

	prev_delta_w := make([]util.Matrix_t, len(self.w))
	for l := range(prev_delta_w) {
		prev_delta_w[l] = util.NewMatrix(len(self.w[l]), len(self.w[l][0]))
	}
	prev_delta_b := make(util.Matrix_t, len(self.b))
	for l := range(prev_delta_b) {
		prev_delta_b[l] = make(util.Row_t, len(self.b[l]))
	}

	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
		fmt.Println("Iteration #", iteration + 1)
		dw, db := self.createDeltaWeights()

		for k := range(trainSet) {
			input := trainSet[k].Input
			output := trainSet[k].Output

			activations := self.forward(input)
			deltas := self.backward(activations, output)
			self.computeDeltaWeights(activations, deltas, dw, db)
		}
		self.applyDeltaWeights(len(trainSet), dw, db, prev_delta_w, prev_delta_b)
	}
//	pretty.Println(self.w)
//	pretty.Println(self.b)
}

func (self *FFNet) Predict(input []float64) []float64 {
	res := self.forward(input)
	return res[len(res) - 1]
}

func (self *FFNet) forward(input []float64) [][]float64 {
	activations := make([][]float64, len(self.w) + 1)
	activations[0] = input
	for l := range(self.w) {
		activations[l+1] = self.computeActivation(self.b[l], input, self.w[l])
		input = activations[l+1]
	}
	return activations
}

func (self *FFNet) backward(activations [][]float64, expected []float64) [][]float64 {
	L := len(self.conf.Layers)
	deltas := make([][]float64, L)
	for l := L - 1; l >= 1; l-- {
		if (l == L - 1) {
			deltas[l] = make([]float64, self.conf.Layers[l])
			for j := range(deltas[l]) {
				deltas[l][j] = (activations[l][j] - expected[j]) * activations[l][j] * (1 - activations[l][j])
			}
		} else {
			deltas[l] = self.computeDelta(activations[l], self.w[l], self.b[l], deltas[l+1])
		}
	}
	return deltas
}


func (self *FFNet) computeActivation(bias util.Row_t, input []float64, w util.Matrix_t) []float64 {
	w_rows := len(w)
	w_cols := len(w[0])

	if len(input) != w_rows {
		panic(fmt.Sprintf("Incomatible sizes: 1x%d * %dx%d", len(input), w_rows, w_cols))
	}
	res := make([]float64, w_cols)

	for j := range(res) {
		s := 1.0 * bias[j]
		for i := 0; i < w_rows; i++ {
			s += float64(input[i]) * w[i][j]
		}
		res[j] = float64(self.conf.Activation(s))
	}

	return res
}

func (self *FFNet) computeDelta(layerActivation []float64, w util.Matrix_t, bias util.Row_t, nextDelta []float64) []float64 {
	w_rows := len(w)
	w_cols := len(w[0])

	if w_cols != len(nextDelta) {
		panic(fmt.Sprintf("Incomatible sizes: %dx%d * %dx1", w_rows, w_cols, len(nextDelta)))
	}
	delta := make([]float64, w_rows)

	for i := range(delta) {
		s := 0.0
		for j := 0; j < w_cols; j++ {
			s += w[i][j] * nextDelta[j]
		}
		delta[i] = s * layerActivation[i] * (1 - layerActivation[i])
	}

	return delta
}

func (self *FFNet) createDeltaWeights() ([]util.Matrix_t, util.Matrix_t) {
	dw := make([]util.Matrix_t, len(self.w))
	for l := range(dw) {
		dw[l] = util.NewMatrix(len(self.w[l]), len(self.w[l][0]))
	}
	db := make(util.Matrix_t, len(self.b))
	for l := range(db) {
		db[l] = make(util.Row_t, len(self.b[l]))
	}

	return dw, db
}

func (self *FFNet) computeDeltaWeights(activations [][]float64, deltas [][]float64, dw []util.Matrix_t, db util.Matrix_t) {
	for l := range(dw) {
		for j := range(db[l]) {
			db[l][j] += deltas[l+1][j] * 1.0
		}
		for i := range(dw[l]) {
			for j := range(dw[l][i]) {
				dw[l][i][j] += deltas[l+1][j] * activations[l][i]
			}
		}
	}
}

func (self *FFNet) applyDeltaWeights(trainSetSize int, dw []util.Matrix_t, db util.Matrix_t, prev_delta_w []util.Matrix_t, prev_delta_b util.Matrix_t) {
	m := float64(trainSetSize)
	for l := range(dw) {
		for j := range(dw[l][0]) {  // swapped loops for bias calculation optimization
			if self.conf.Bias {
				delta_b_lj := - self.conf.LearningRate * db[l][j] / m + self.conf.Momentum * prev_delta_b[l][j]
				self.b[l][j] += delta_b_lj
				prev_delta_b[l][j] = delta_b_lj
			}
			for i := range(dw[l]) {
				reg := self.conf.Regularization * self.w[l][i][j]
				delta_w_lij := - self.conf.LearningRate * (dw[l][i][j] / m + reg) + self.conf.Momentum * prev_delta_w[l][i][j]
				self.w[l][i][j] += delta_w_lij
				prev_delta_w[l][i][j] = delta_w_lij
			}
		}
	}
}