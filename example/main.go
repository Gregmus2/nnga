package main

import (
	"github.com/Gregmus2/nnga"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"log"
	"math"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	ga := nnga.NewGA(1000, &deep.Config{
		/* Input dimensionality */
		Inputs: 2,
		/* Two hidden layers consisting of two neurons each, and a single output */
		Layout: []int{2, 2, 1},
		/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
		Activation: deep.ActivationSigmoid,
		/* Determines output layer activation & loss function:
		ModeRegression: linear outputs with MSE loss
		ModeMultiClass: softmax output with Cross Entropy loss
		ModeMultiLabel: sigmoid output with Cross Entropy loss
		ModeBinary: sigmoid output with binary CE loss */
		Mode: deep.ModeBinary,
		/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
		Weight: deep.NewNormal(1.0, 0.0),
		/* Apply bias */
		Bias: true,
	}, &nnga.Coefficients{
		Scale:                   1,
		Selection:               0.2,
		MutationClassic:         0.1,
		MutationGrowth:          2,
		MutationGenesMaxPercent: 0.2,
		MutationOffset:          0.1,
	})

	data := training.Examples{
		{[]float64{1, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{0, 0}, []float64{1}},
	}
	count := float64(len(data))

	min := 1.0
	i := 0
	for ; ; i++ {
		for _, person := range ga.Persons {
			scoreSum := 0.0
			diffSum := 0.0
			for _, example := range data {
				res := person.Predict(example.Input)
				diff := math.Abs(example.Response[0] - res[0])
				scoreSum += 1 - diff
				diffSum += diff
			}
			avgDiff := diffSum / count
			if avgDiff < min {
				min = avgDiff
			}
			person.Score(scoreSum / count)
		}

		if min < 0.000001 {
			break
		}

		ga.Evolve()
	}

	log.Printf("iterations: %d", i)
}
