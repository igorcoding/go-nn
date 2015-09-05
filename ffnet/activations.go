package ffnet
import "math"

type ActivationFunc func(float64) float64

func Sigma(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}