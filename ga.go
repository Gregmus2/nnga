package main

import (
	"errors"
	"github.com/gonum/stat"
	"github.com/patrikeh/go-deep"
	"math"
	"math/rand"
	"sort"
)

type GA struct {
	Persons    []*Person
	Cfg        *deep.Config
	Coeffs     *Coefficients
	genesCount int
}

type Coefficients struct {
	scale                float64 // 1-5
	selection            float64 // % of survivors
	crossbreeding        float64 // % of children / 2
	mutationClassic      float64 // % of mutations by classic method
	mutationGrowth       float64 // 1.0 = -0.5 - 0.5
	mutationGenesPercent float64 // max percent of genes will be mutate
	mutationOffset       float64 // % of mutations by offset method
}

// size should be sqr of int
func NewGA(size int, cfg *deep.Config, coeffs *Coefficients) *GA {
	persons := make([]*Person, size)
	for i := 0; i < size; i++ {
		persons[i] = NewPerson(cfg)
	}

	inputWeightsCount := (cfg.Inputs + 1) * cfg.Layout[0]
	hiddenLayersWeightsCount := (cfg.Layout[0] + 1) * cfg.Layout[1]
	outputWeightsCount := (cfg.Layout[1] + 1) * cfg.Layout[2]

	return &GA{
		Persons:    persons,
		Cfg:        cfg,
		Coeffs:     coeffs,
		genesCount: inputWeightsCount + hiddenLayersWeightsCount + outputWeightsCount,
	}
}

func (ga *GA) Evolve() {
	ga.selection()
	ga.crossover()
	ga.mutation()

	for _, person := range ga.Persons {
		person.Reborn()
	}
}

func (ga *GA) rouletteSelection() error {
	scores := make([]float64, len(ga.Persons))
	for i, person := range ga.Persons {
		if person.scored == false {
			return errors.New("not all persons was scored")
		}

		scores[i] = person.score
	}
	mean, std := stat.MeanStdDev(scores, nil)

	sumOfScaledScores := 0.0
	scaledScores := make([]float64, len(ga.Persons))
	for i, person := range ga.Persons {
		scaledScores[i] = person.score + (mean - ga.Coeffs.scale*std)
		sumOfScaledScores += scaledScores[i]
	}

	chances := make([]float64, len(scaledScores))
	for i := 0; i < len(scaledScores); i++ {
		chances[i] = scaledScores[i] / sumOfScaledScores
	}

	newGenerationCount := int(math.Round(ga.Coeffs.selection * float64(len(ga.Persons))))
	newGeneration := make([]*Person, 0, newGenerationCount)
	maxValue := 1.0
	for i := 0; i < newGenerationCount; i++ {
		sum := 0.0
		x := rand.Float64() * maxValue
		for i := 0; i < len(chances); i++ {
			sum += chances[i]
			if x <= sum {
				newGeneration = append(newGeneration, ga.Persons[i])
				maxValue -= chances[i]
				chances[i] = 0
				break
			}
		}
	}

	ga.Persons = newGeneration

	return nil
}

func (ga *GA) selection() {
	newGenerationCount := int(math.Round(ga.Coeffs.selection * float64(len(ga.Persons))))
	sort.Slice(ga.Persons, func(i, j int) bool {
		return ga.Persons[i].score > ga.Persons[j].score
	})

	ga.Persons = ga.Persons[:newGenerationCount]
}

func (ga *GA) crossover() {
	populationSize := len(ga.Persons)
	count := math.Round(ga.Coeffs.crossbreeding * float64(populationSize))

	for i := 0; i < int(count); i++ {
		mother := ga.Persons[rand.Intn(populationSize)]
		father := ga.Persons[rand.Intn(populationSize)]
		cutPoint := rand.Intn(ga.genesCount-2) + 1
		child1 := NewPerson(ga.Cfg)
		ga.Persons = append(ga.Persons, child1)
		child2 := NewPerson(ga.Cfg)
		ga.Persons = append(ga.Persons, child2)

		copy(child1.weights[:cutPoint], mother.weights[:cutPoint])
		copy(child1.weights[cutPoint:], father.weights[cutPoint:])
		copy(child2.weights[:cutPoint], father.weights[:cutPoint])
		copy(child2.weights[cutPoint:], mother.weights[cutPoint:])
	}
}

func (ga *GA) mutation() {
	Kc := ga.Coeffs.mutationClassic
	Kg := ga.Coeffs.mutationGrowth
	Kgc := ga.Coeffs.mutationGenesPercent
	Ko := ga.Coeffs.mutationOffset
	populationSize := len(ga.Persons)
	mutatedGenesMax := int(math.Round(Kgc * float64(ga.genesCount)))

	mutantsCount := int(math.Round(Kc * float64(populationSize)))
	for i := 0; i < mutantsCount; i++ {
		growth := (rand.Float64() - 0.5) * Kg
		person := ga.Persons[rand.Intn(populationSize)]
		for j := 0; j < rand.Intn(mutatedGenesMax); j++ {
			person.weights[rand.Intn(person.genesCount)].value += growth
		}
	}

	mutantsCount = int(math.Round(Ko * float64(populationSize)))
	for i := 0; i < mutantsCount; i++ {
		person := ga.Persons[rand.Intn(populationSize)]
		for j := 0; j < rand.Intn(mutatedGenesMax); j++ {
			target := rand.Intn(person.genesCount - 1)
			neighbour := rand.Intn(person.genesCount - 1)
			x := person.weights[target].value
			person.weights[target].value = person.weights[neighbour].value
			person.weights[neighbour].value = x
		}
	}
}
