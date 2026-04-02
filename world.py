import json
import os
import numpy as np
import config
from fish import Fish, push_fish_apart
from food import FoodManager
from genes import Genes
from neural_network import NeuralNetwork


class HallOfFame:
    """Keeps track of the best fish ever seen."""

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.entries = []  # list of (total_food, brain, genes) tuples

    def consider(self, fish):
        """Maybe add this fish to hall of fame."""
        score = fish.total_food
        if len(self.entries) < self.max_size:
            self.entries.append((score, fish.brain.clone(), fish.genes.clone()))
            self.entries.sort(key=lambda e: e[0], reverse=True)
        elif score > self.entries[-1][0]:
            self.entries[-1] = (score, fish.brain.clone(), fish.genes.clone())
            self.entries.sort(key=lambda e: e[0], reverse=True)

    def spawn_champions(self, count=None):
        """Create fish from hall of fame entries with mutations, near oases."""
        from food import get_oases

        if not self.entries:
            return [Fish(genes=Genes.random()) for _ in range(count or config.MIN_POPULATION)]

        count = count or len(self.entries)
        oases = get_oases()
        fishes = []
        for i in range(count):
            entry = self.entries[i % len(self.entries)]
            _, brain, genes = entry
            f = Fish(genes=genes.clone())
            f.brain = brain.clone()
            # Light mutation so they're not exact copies
            NeuralNetwork.mutate(f.brain, config.MUTATION_RATE * 0.5, config.MUTATION_STRENGTH * 0.5)
            Genes.mutate(f.genes, rate=0.1, strength=0.05)
            # Spawn near a random oasis so they can find food
            if oases:
                oidx = i % len(oases)
                f.x = oases[oidx][0] + np.random.uniform(-300, 300)
                f.y = oases[oidx][1] + np.random.uniform(-300, 300)
            fishes.append(f)
        return fishes


class World:
    def __init__(self):
        self.food = FoodManager()
        self.population = []
        self.hall_of_fame = HallOfFame(config.HALL_OF_FAME_SIZE)
        self.tick_count = 0
        self.total_births = 0
        self.total_deaths = 0
        self.extinctions = 0

        # Stats
        self.selected_fish = None
        self.best_ever_food = 0
        self.alive_peak = 0

        self.reset()

    def reset(self):
        from food import get_oases
        self.population = []
        # Create population from a single gene pool, spawned near oases
        base = Genes.random()
        tribe_genes = Genes.random_tribe(base=base, count=config.POPULATION_SIZE, spread=0.05)
        oases = get_oases()
        for i, genes in enumerate(tribe_genes):
            f = Fish(genes=genes)
            # Place near oases so they start close to food
            if oases:
                oidx = i % len(oases)
                f.x = oases[oidx][0] + np.random.uniform(-200, 200)
                f.y = oases[oidx][1] + np.random.uniform(-200, 200)
            self.population.append(f)
        self.food.reset()
        self.tick_count = 0
        self.total_births = 0
        self.total_deaths = 0
        self.extinctions = 0
        self.hall_of_fame = HallOfFame(config.HALL_OF_FAME_SIZE)
        self.selected_fish = None
        self.best_ever_food = 0
        self.alive_peak = 0

    def tick(self):
        """One simulation step."""
        food_items = self.food.items

        # Update alive fish
        for fish in self.population:
            if fish.alive:
                fish.update(food_items, self.population)

        # Collisions
        push_fish_apart(self.population)
        self.food.push_food_from_fish(self.population)
        self.food.update()
        self.food.check_eating(self.population)

        # --- Reproduction (sexual) ---
        new_fish = []
        already_mated = set()
        for fish in self.population:
            if not fish.can_reproduce or id(fish) in already_mated:
                continue
            if self.alive_count + len(new_fish) >= config.MAX_POPULATION:
                break
            mate = fish.find_mate(self.population)
            if mate is None or id(mate) in already_mated:
                continue
            child = fish.reproduce(mate)
            if child:
                new_fish.append(child)
                self.total_births += 1
                already_mated.add(id(fish))
                already_mated.add(id(mate))
        self.population.extend(new_fish)

        # --- Death cleanup + hall of fame ---
        still_alive = []
        for fish in self.population:
            if fish.alive:
                still_alive.append(fish)
            else:
                # Consider for hall of fame before removing
                self.hall_of_fame.consider(fish)
                self.total_deaths += 1
                if fish.total_food > self.best_ever_food:
                    self.best_ever_food = fish.total_food
                # Deselect if selected fish died
                if fish is self.selected_fish:
                    self.selected_fish = None
        self.population = still_alive

        # --- Extinction check ---
        if len(self.population) < config.MIN_POPULATION:
            self.extinctions += 1
            champions = self.hall_of_fame.spawn_champions(config.POPULATION_SIZE)
            self.population.extend(champions)

        # Stats
        ac = self.alive_count
        if ac > self.alive_peak:
            self.alive_peak = ac
        # Track best ever including alive fish
        baf = self.best_alive_food
        if baf > self.best_ever_food:
            self.best_ever_food = baf

        self.tick_count += 1

    @property
    def alive_count(self):
        return len(self.population)

    @property
    def avg_food(self):
        if not self.population:
            return 0
        return sum(f.total_food for f in self.population) / len(self.population)

    @property
    def best_alive_food(self):
        if not self.population:
            return 0
        return max(f.total_food for f in self.population)

    @property
    def avg_size(self):
        if not self.population:
            return 1.0
        return sum(f.genes.size for f in self.population) / len(self.population)

    @property
    def time_str(self):
        """Formatted elapsed time."""
        secs = self.tick_count // config.FPS
        mins = secs // 60
        secs = secs % 60
        return f"{mins}:{secs:02d}"

    def save_fish(self, fish, path="saves"):
        os.makedirs(path, exist_ok=True)
        data = {
            "topology": fish.brain.topology,
            "recurrent_size": fish.brain.recurrent_size,
            "weights": fish.brain.get_flat_weights().tolist(),
            "genes": fish.genes.to_dict(),
            "total_food": fish.total_food,
        }
        filename = os.path.join(path, f"fish_t{self.tick_count}_{id(fish) % 10000}.json")
        with open(filename, "w") as f:
            json.dump(data, f)
        return filename

    def load_fish(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        genes = Genes.from_dict(data["genes"]) if "genes" in data else Genes()
        fish = Fish(genes=genes)
        topo = data.get("topology", config.NN_TOPOLOGY)
        rec = data.get("recurrent_size", 0)
        fish.brain = NeuralNetwork(topo, recurrent_size=rec)
        fish.brain.set_flat_weights(np.array(data["weights"]))
        return fish

    def _fish_to_dict(self, fish):
        """Serialize a fish (brain + genes + stats)."""
        return {
            "topology": fish.brain.topology,
            "recurrent_size": fish.brain.recurrent_size,
            "weights": fish.brain.get_flat_weights().tolist(),
            "genes": fish.genes.to_dict(),
            "total_food": fish.total_food,
            "x": fish.x,
            "y": fish.y,
        }

    def _fish_from_dict(self, d):
        """Deserialize a fish."""
        genes = Genes.from_dict(d["genes"])
        f = Fish(x=d.get("x"), y=d.get("y"), genes=genes)
        # Reconstruct brain with saved topology (handles old saves too)
        topo = d.get("topology", config.NN_TOPOLOGY)
        rec = d.get("recurrent_size", 0)
        f.brain = NeuralNetwork(topo, recurrent_size=rec)
        f.brain.set_flat_weights(np.array(d["weights"]))
        f.total_food = d.get("total_food", 0)
        return f

    def save_world(self, path="saves"):
        """Save entire population + hall of fame + stats."""
        os.makedirs(path, exist_ok=True)
        data = {
            "tick_count": self.tick_count,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "extinctions": self.extinctions,
            "best_ever_food": self.best_ever_food,
            "alive_peak": self.alive_peak,
            "population": [self._fish_to_dict(f) for f in self.population if f.alive],
            "hall_of_fame": [
                {"score": score, "topology": brain.topology,
                 "recurrent_size": brain.recurrent_size,
                 "weights": brain.get_flat_weights().tolist(),
                 "genes": genes.to_dict()}
                for score, brain, genes in self.hall_of_fame.entries
            ],
        }
        filename = os.path.join(path, "world_save.json")
        with open(filename, "w") as f:
            json.dump(data, f)
        return filename

    def load_world(self, filename):
        """Load entire population + hall of fame + stats."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Restore stats
        self.tick_count = data.get("tick_count", 0)
        self.total_births = data.get("total_births", 0)
        self.total_deaths = data.get("total_deaths", 0)
        self.extinctions = data.get("extinctions", 0)
        self.best_ever_food = data.get("best_ever_food", 0)
        self.alive_peak = data.get("alive_peak", 0)

        # Restore population
        self.population = []
        for fd in data.get("population", []):
            self.population.append(self._fish_from_dict(fd))

        # Restore hall of fame
        self.hall_of_fame = HallOfFame(config.HALL_OF_FAME_SIZE)
        for entry in data.get("hall_of_fame", []):
            from neural_network import NeuralNetwork as NN
            topo = entry.get("topology", config.NN_TOPOLOGY)
            rec = entry.get("recurrent_size", 0)
            brain = NN(topo, recurrent_size=rec)
            brain.set_flat_weights(np.array(entry["weights"]))
            genes = Genes.from_dict(entry["genes"])
            self.hall_of_fame.entries.append((entry["score"], brain, genes))

        # Reset food
        self.food.reset()
        self.selected_fish = None
