"""Pre-training arena: small 600x600 world with generational evolution.

Fish learn to eat here through fast generational selection:
- Each generation runs for GEN_TIME seconds
- Top ELITE fish survive, rest are replaced with mutated offspring
- No sexual reproduction needed — pure neuroevolution
"""
import json
import os
import math
import numpy as np
import config
from fish import Fish
from genes import Genes
from neural_network import NeuralNetwork
from food import FoodManager

TRAINING_SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_genomes.json")

# Config overrides for the training arena
TRAINING_OVERRIDES = {
    'WORLD_WIDTH': 600,
    'WORLD_HEIGHT': 600,
    'OASIS_COUNT': 1,
    'OASIS_RADIUS_W': 250,
    'OASIS_RADIUS_H': 250,
    'FOOD_COUNT': 10,
}

# Training-specific params (not in config)
TRAIN_POP = 10
TRAIN_GEN_TICKS = 900   # 15 sec at 60fps
TRAIN_ELITE = 2
TRAIN_TOURNAMENT_SIZE = 3
TRAIN_MUTATION_RATE = 0.12
TRAIN_MUTATION_STRENGTH = 0.35


class TrainingWorld:
    """Simplified world for generational training."""

    def __init__(self):
        self.food = FoodManager()
        self.food.reset()
        self.population = []
        self.generation = 0
        self.gen_tick = 0
        self.tick_count = 0
        self.best_ever_food = 0
        self.best_gen_food = 0

        # Stats history
        self.gen_best_history = []

        self._init_population()

    def _init_population(self):
        self.population = []
        base_genes = Genes.random()
        for _ in range(TRAIN_POP):
            genes = base_genes.clone()
            Genes.mutate(genes, rate=0.3, strength=0.15)
            self.population.append(Fish(genes=genes))

    def tick(self):
        food_items = self.food.items

        for fish in self.population:
            if fish.alive:
                fish.update(food_items, self.population)

        from fish import push_fish_apart
        push_fish_apart(self.population)
        self.food.push_food_from_fish(self.population)
        self.food.update()
        self.food.check_eating(self.population)

        self.gen_tick += 1
        self.tick_count += 1

        # Track best
        for fish in self.population:
            if fish.total_food > self.best_ever_food:
                self.best_ever_food = fish.total_food

        # Generation end
        if self.gen_tick >= TRAIN_GEN_TICKS:
            self._next_generation()

    def _fitness(self, fish):
        """Combined fitness: food is primary, distance is secondary (helps early learning)."""
        # Distance bonus scaled down so 1 food > ~300px travel
        dist_bonus = fish.distance_traveled / 1000.0
        return fish.total_food * 10.0 + dist_bonus

    def _tournament_select(self, ranked_with_fitness):
        """Tournament selection: pick TOURNAMENT_SIZE random, return the best."""
        contestants = [ranked_with_fitness[np.random.randint(len(ranked_with_fitness))]
                       for _ in range(TRAIN_TOURNAMENT_SIZE)]
        return max(contestants, key=lambda x: x[0])

    def _next_generation(self):
        # Calculate fitness for each fish
        scored = [(self._fitness(f), f) for f in self.population]
        scored.sort(key=lambda x: x[0], reverse=True)

        self.best_gen_food = scored[0][1].total_food if scored else 0
        self.gen_best_history.append(self.best_gen_food)

        # Keep elite brains + genes
        elites = []
        for _, fish in scored[:TRAIN_ELITE]:
            elites.append((fish.brain.clone(), fish.genes.clone()))

        # Create new population
        new_pop = []
        for i in range(TRAIN_POP):
            if i < TRAIN_ELITE:
                # Elite: exact copy
                brain, genes = elites[i]
                f = Fish(genes=genes.clone())
                f.brain = brain.clone()
            else:
                # Asymmetric reproduction: brain from BETTER parent + mutation
                # (no brain crossover - avoids permutation problem)
                s1, p1 = self._tournament_select(scored)
                s2, p2 = self._tournament_select(scored)
                better = p1 if s1 >= s2 else p2
                f = Fish(genes=Genes.crossover(p1.genes, p2.genes))
                f.brain = better.brain.clone()
                # Adaptive mutation: fitter parent → gentler mutation
                fit_factor = max(0.5, 1.0 - better.total_food / 15.0)
                NeuralNetwork.mutate(f.brain,
                                     TRAIN_MUTATION_RATE * fit_factor,
                                     TRAIN_MUTATION_STRENGTH * fit_factor)
                f.brain.reset_state()
                Genes.mutate(f.genes, rate=0.1, strength=0.05)

            new_pop.append(f)

        self.population = new_pop
        self.food.reset()
        self.generation += 1
        self.gen_tick = 0

    @property
    def alive_count(self):
        return sum(1 for f in self.population if f.alive)

    @property
    def best_alive_food(self):
        alive = [f for f in self.population if f.alive]
        return max((f.total_food for f in alive), default=0)

    @property
    def time_str(self):
        secs = self.tick_count // 60
        mins = secs // 60
        secs = secs % 60
        return f"{mins}:{secs:02d}"

    @property
    def gen_progress(self):
        """0.0 - 1.0 progress through current generation."""
        return self.gen_tick / TRAIN_GEN_TICKS


class TrainingArena:
    def __init__(self):
        self.active = False
        self.world = None
        self._saved_config = {}

    def enter(self):
        """Switch to training arena."""
        self._saved_config = {}
        for key, val in TRAINING_OVERRIDES.items():
            self._saved_config[key] = getattr(config, key)
            setattr(config, key, val)

        self.world = TrainingWorld()
        self.active = True

    def exit(self):
        """Leave training arena, restore main config."""
        for key, val in self._saved_config.items():
            setattr(config, key, val)
        self.active = False
        self.world = None

    def reset(self):
        """Reset the training world."""
        if self.active:
            self.world = TrainingWorld()

    def save_top_genomes(self, count=10):
        """Save the best fish from training."""
        if not self.world:
            return 0

        ranked = sorted(self.world.population, key=lambda f: f.total_food, reverse=True)

        data = []
        for fish in ranked[:count]:
            data.append({
                'total_food': int(fish.total_food),
                'topology': fish.brain.topology,
                'recurrent_size': fish.brain.recurrent_size,
                'weights': fish.brain.get_flat_weights().tolist(),
                'genes': fish.genes.to_dict(),
            })

        if not data:
            return 0

        with open(TRAINING_SAVE_FILE, 'w') as f:
            json.dump(data, f)

        return len(data)

    @staticmethod
    def has_trained_genomes():
        return os.path.exists(TRAINING_SAVE_FILE)

    @staticmethod
    def load_trained_fish(main_world):
        """Load trained genomes and add them to the main world population."""
        if not os.path.exists(TRAINING_SAVE_FILE):
            return 0

        with open(TRAINING_SAVE_FILE, 'r') as f:
            data = json.load(f)

        count = 0
        for entry in data:
            genes = Genes.from_dict(entry['genes'])
            fish = Fish(genes=genes)
            from neural_network import NeuralNetwork
            topo = entry.get('topology', config.NN_TOPOLOGY)
            rec = entry.get('recurrent_size', 0)
            fish.brain = NeuralNetwork(topo, recurrent_size=rec)
            fish.brain.set_flat_weights(np.array(entry['weights']))
            # Place near a random oasis
            from food import get_oases
            oases = get_oases()
            if oases:
                oidx = np.random.randint(len(oases))
                fish.x = oases[oidx][0] + np.random.uniform(-100, 100)
                fish.y = oases[oidx][1] + np.random.uniform(-100, 100)

            if len(main_world.population) >= config.MAX_POPULATION:
                worst = min(main_world.population, key=lambda f: f.total_food)
                idx = main_world.population.index(worst)
                main_world.population[idx] = fish
            else:
                main_world.population.append(fish)
            count += 1

        return count
