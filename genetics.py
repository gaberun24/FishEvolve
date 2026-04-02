import numpy as np
import config
from fish import Fish
from neural_network import NeuralNetwork


class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0

    def create_population(self):
        """Create initial random population."""
        return [Fish() for _ in range(config.POPULATION_SIZE)]

    def tournament_select(self, population, k=3):
        """Select one parent via tournament selection."""
        candidates = np.random.choice(len(population), size=k, replace=False)
        best = max(candidates, key=lambda i: population[i].fitness)
        return population[best]

    def evolve(self, population):
        """Create next generation from current population."""
        pop_size = config.POPULATION_SIZE
        elitism = config.ELITISM_COUNT

        for fish in population:
            fish.calculate_fitness()

        ranked = sorted(population, key=lambda f: f.fitness, reverse=True)

        best_fitness = ranked[0].fitness
        avg_fitness = sum(f.fitness for f in ranked) / len(ranked)
        best_food = ranked[0].food_eaten

        new_population = []

        # Elitism: copy top performers directly
        for i in range(min(elitism, len(ranked))):
            elite = Fish()
            elite.brain = ranked[i].brain.clone()
            new_population.append(elite)

        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent_a = self.tournament_select(ranked)
            parent_b = self.tournament_select(ranked)
            child_brain = NeuralNetwork.crossover(parent_a.brain, parent_b.brain)
            NeuralNetwork.mutate(child_brain, config.MUTATION_RATE, config.MUTATION_STRENGTH)
            child = Fish()
            child.brain = child_brain
            new_population.append(child)

        self.generation += 1

        return new_population, best_fitness, avg_fitness, best_food
