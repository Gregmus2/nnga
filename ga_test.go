package nnga

import (
	"github.com/patrikeh/go-deep"
	"math"
	"math/rand"
	"testing"
)

var defaultNeuralCfg = &deep.Config{
	Inputs:     1,
	Layout:     []int{1, 1, 1},
	Activation: deep.ActivationSigmoid,
	Mode:       deep.ModeBinary,
	Bias:       true,
}

func TestNewGA(t *testing.T) {
	ga := NewGA(1000, defaultNeuralCfg, &Coefficients{Selection: 0.2, Scale: 3})
	if len(ga.Persons) != 1000 {
		t.Fatalf("wrong amount of persons on the beginning: %d", len(ga.Persons))
	}
}

func TestGA_selection(t *testing.T) {
	ga := NewGA(1000, defaultNeuralCfg, &Coefficients{Selection: 0.2, Scale: 3})
	for _, person := range ga.Persons {
		person.Score(-(rand.Float64() * 10))
	}

	ga.selection()
	if len(ga.Persons) != 1000*0.2 {
		t.Fatalf("wrong amount of persons after selection: %d", len(ga.Persons))
	}
}

func TestGA_crossover(t *testing.T) {
	ga := NewGA(1000, defaultNeuralCfg, &Coefficients{Selection: 0.2, Scale: 3})
	for _, person := range ga.Persons {
		person.Score(-(rand.Float64() * 10))
	}

	populationSize := float64(len(ga.Persons))
	ga.crossover()
	if len(ga.Persons) != 1000 {
		t.Fatalf("wrong amount of persons after crossover: %d != %d", len(ga.Persons), 1000)
	}
}
