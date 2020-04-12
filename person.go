package nnga

import "github.com/patrikeh/go-deep"

type Weight struct {
	layer  int
	neuron int
	input  int
	value  float64
}

type Person struct {
	n          *deep.Neural
	weights    []Weight
	score      float64
	scored     bool
	genesCount int
}

func NewPerson(cfg *deep.Config) *Person {
	n := deep.NewNeural(cfg)
	inputWeightsCount := (cfg.Inputs + 1) * cfg.Layout[0]
	hiddenLayersWeightsCount := (cfg.Layout[0] + 1) * cfg.Layout[1]
	outputWeightsCount := (cfg.Layout[1] + 1) * cfg.Layout[2]
	genesCount := inputWeightsCount + hiddenLayersWeightsCount + outputWeightsCount
	p := &Person{
		n:          n,
		genesCount: inputWeightsCount + hiddenLayersWeightsCount + outputWeightsCount,
		weights:    make([]Weight, genesCount),
	}

	w := n.Weights()
	j := 0
	for layer, layers := range w {
		for neuron, neurons := range layers {
			for input, weight := range neurons {
				p.weights[j] = Weight{layer: layer, neuron: neuron, input: input, value: weight}
				j++
			}
		}
	}

	return p
}

func (p *Person) Reborn() {
	w := p.n.Weights()
	for _, weight := range p.weights {
		w[weight.layer][weight.neuron][weight.input] = weight.value
	}
	p.n.ApplyWeights(w)
}

func (p *Person) RebornNative(weights [][][]float64) {
	p.n.ApplyWeights(weights)
	p.scored = false
}

func (p *Person) Predict(input []float64) []float64 {
	return p.n.Predict(input)
}

func (p *Person) Score(score float64) {
	p.score = score
	p.scored = true
}
