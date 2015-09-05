package ffnet
import (
	"fmt"
	"math/rand"
	"time"
    "github.com/kr/pretty"
)

var _ = fmt.Printf

type FFNetConf struct {
	Layers []int32
	Alpha float64
	Eta float64
	Regularization float64
	Bias bool
	Iterations int
	Activation ActivationFunc
}

type ffNet struct {
	conf *FFNetConf
	w []matrix_t
	b row_t

	randomSource *rand.Rand
}

type TrainExample struct {
	Input []float64
	Output []float64
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
				nn.w[l][i][j] = nn.randomSource.Float64()
			}
		}
	}
	nn.b = make(row_t, len(conf.Layers) - 1)
	for l := range(nn.b) {
		if conf.Bias {
			nn.b[l] = nn.randomSource.Float64()
		} else {
			nn.b[l] = 0
		}
	}
	return nn
}

func (self *ffNet) Train(trainSet []TrainExample) {

	for iteration := 0; iteration < self.conf.Iterations; iteration++ {
		fmt.Println("Iteration #", iteration + 1)
		delta_w := make([]matrix_t, len(self.w))
		for l := range(delta_w) {
			delta_w[l] = make([]row_t, len(self.w[l]))
			for i := range(delta_w[l]) {
				delta_w[l][i] = make(row_t, len(self.w[l][i]))
			}
		}

		m := float64(len(trainSet))
		for k := range(trainSet) {
			input := trainSet[k].Input
			activations := self.forward(input)
			deltas := self.backward(activations, trainSet[k].Output)

			for l := range(delta_w) {
				for i := range(delta_w[l]) {
					for j := range(delta_w[l][i]) {
						delta_w[l][i][j] += deltas[l+1][j] * activations[l][i]
					}
				}
			}
		}
		for l := range(delta_w) {
			for i := range(delta_w[l]) {
				for j := range(delta_w[l][i]) {
					self.w[l][i][j] -= self.conf.Alpha * delta_w[l][i][j] / m
				}
			}
		}
		pretty.Println(self.w)
	}
}

func (self *ffNet) Predict(input []float64) []float64 {
	res := self.forward(input)
	return res[len(res) - 1]
}

func (self *ffNet) forward(input []float64) [][]float64 {
	activations := make([][]float64, len(self.w) + 1)
	activations[0] = input
	for l := range(self.w) {
		activations[l+1] = self.computeActivation(l, input, self.w[l])
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
			deltas[l] = self.computeDelta(activations[l], self.w[l], deltas[l+1])
		}
	}
	return deltas
}


func (self *ffNet) computeActivation(layer int, input []float64, w matrix_t) []float64 {
	b_rows := len(w)
	b_cols := len(w[0])

	if len(input) != b_rows {
		panic(fmt.Sprintf("Incomatible sizes: 1x%d * %dx%d", len(input), b_rows, b_cols))
	}
	res := make([]float64, b_cols)

	for j := range(res) {
		s := 1.0 * self.b[layer]
		for i := 0; i < b_rows; i++ {
			s += float64(input[i]) * w[i][j]
		}
		res[j] = float64(self.conf.Activation(s))
	}

	return res
}

func (self *ffNet) computeDelta(layerActivation []float64, w matrix_t, nextDelta []float64) []float64 {
	w_rows := len(w)
	w_cols := len(w[0])

	if w_cols != len(nextDelta) {
		panic(fmt.Sprintf("Incomatible sizes: %dx%d * %dx1", w_rows, w_cols, len(nextDelta)))
	}
	res := make([]float64, w_rows)

	for i := range(res) {
		s := 0.0
		for j := 0; j < w_cols; j++ {
			s += w[i][j] * nextDelta[j]
		}
		res[i] = s * layerActivation[i] * (1 - layerActivation[i])
	}

	return res
}
