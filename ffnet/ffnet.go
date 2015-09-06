package ffnet
import (
	"fmt"
	"math/rand"
	"time"
	"github.com/kr/pretty"
)

var _ = fmt.Printf

const (
	RAND_EPSILON = 0.5
)

type FFNetConf struct {
	Layers []int32
	Alpha float64
	Regularization float64
	Bias bool
	Iterations int
	Activation ActivationFunc
}

type ffNet struct {
	conf *FFNetConf
	w []matrix_t
	b matrix_t

	randomSource *rand.Rand
}

type TrainExample struct {
	Input []float64
	Output []float64
}

func NewTrainExample(input, output []float64) TrainExample {
	return TrainExample{Input:input, Output:output}
}

func BuildFFNet(conf *FFNetConf) *ffNet {
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

	nn := &ffNet{conf:conf}
	nn.randomSource = rand.New(rand.NewSource(time.Now().UnixNano()))

	nn.w = make([]matrix_t, len(conf.Layers) - 1);
	for l := range(nn.w) {
		nn.w[l] = make(matrix_t, conf.Layers[l])
		for i := range(nn.w[l]) {
			nn.w[l][i] = make(row_t, conf.Layers[l + 1])
			for j := range(nn.w[l][i]) {
				nn.w[l][i][j] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
			}
		}
	}
	nn.b = make(matrix_t, len(conf.Layers) - 1)
	for l := range(nn.b) {
		nn.b[l] = make(row_t, conf.Layers[l+1])
		if conf.Bias {
			for j := range(nn.b[l]) {
				nn.b[l][j] = nn.randomSource.Float64() * (2 * RAND_EPSILON) - RAND_EPSILON
			}
		}
	}
	return nn
}

func (self *ffNet) Train(trainSet []TrainExample) {

	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
		fmt.Println("Iteration #", iteration + 1)
		delta_w, delta_b := self.createDeltaWeights()

		for k := range(trainSet) {
			input := trainSet[k].Input
			activations := self.forward(input)
			deltas := self.backward(activations, trainSet[k].Output)
			self.computeDeltaWeights(activations, deltas, delta_w, delta_b)
		}
		self.applyDeltaWeights(len(trainSet), delta_w, delta_b)
	}
	pretty.Println(self.w)
	pretty.Println(self.b)
}

func (self *ffNet) Predict(input []float64) []float64 {
	res := self.forward(input)
	return res[len(res) - 1]
}

func (self *ffNet) forward(input []float64) [][]float64 {
	activations := make([][]float64, len(self.w) + 1)
	activations[0] = input
	for l := range(self.w) {
		activations[l+1] = self.computeActivation(self.b[l], input, self.w[l])
		input = activations[l+1]
	}
	return activations
}

func (self *ffNet) backward(activations [][]float64, expected []float64) [][]float64 {
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


func (self *ffNet) computeActivation(bias row_t, input []float64, w matrix_t) []float64 {
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

func (self *ffNet) computeDelta(layerActivation []float64, w matrix_t, bias row_t, nextDelta []float64) []float64 {
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

func (self *ffNet) createDeltaWeights() ([]matrix_t, matrix_t) {
	delta_w := make([]matrix_t, len(self.w))
	for l := range(delta_w) {
		delta_w[l] = make([]row_t, len(self.w[l]))
		for i := range(delta_w[l]) {
			delta_w[l][i] = make(row_t, len(self.w[l][i]))
		}
	}
	delta_b := make(matrix_t, len(self.b))
	for l := range(delta_b) {
		delta_b[l] = make(row_t, len(self.b[l]))
	}

	return delta_w, delta_b
}

func (self *ffNet) computeDeltaWeights(activations [][]float64, deltas [][]float64, delta_w []matrix_t, delta_b matrix_t) {
	for l := range(delta_w) {
		for j := range(delta_b[l]) {
			delta_b[l][j] += deltas[l+1][j] * 1.0
		}
		for i := range(delta_w[l]) {
			for j := range(delta_w[l][i]) {
				delta_w[l][i][j] += deltas[l+1][j] * activations[l][i]
			}
		}
	}
}

func (self *ffNet) applyDeltaWeights(trainSetSize int, delta_w []matrix_t, delta_b matrix_t) {
	m := float64(trainSetSize)
	for l := range(delta_w) {
		for j := range(delta_w[l][0]) {  // swapped loops for bias calculation optimization
			if self.conf.Bias {
				self.b[l][j] -= self.conf.Alpha * delta_b[l][j] / m
			}
			for i := range(delta_w[l]) {
				self.w[l][i][j] -= self.conf.Alpha * delta_w[l][i][j] / m
			}
		}
	}
}