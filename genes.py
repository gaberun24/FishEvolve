"""Genetic traits that affect fish appearance and physics."""
import numpy as np


class Genes:
    __slots__ = ('size', 'color_r', 'color_g', 'color_b',
                 'fin_size', 'metabolism')

    def __init__(self, size=1.0, color_r=239, color_g=127, color_b=26,
                 fin_size=1.0, metabolism=1.0):
        self.size = size              # 0.5 - 2.0: body scale
        self.color_r = color_r        # 0-255
        self.color_g = color_g
        self.color_b = color_b
        self.fin_size = fin_size      # 0.5 - 1.5: bigger = better turn, more drag
        self.metabolism = metabolism   # 0.5 - 2.0: fast = more speed but more energy burn

    @property
    def color(self):
        return (int(self.color_r), int(self.color_g), int(self.color_b))

    def compatible_with(self, other, threshold=0.10):
        """Check if two gene sets are compatible for mating.
        Returns True if size and color differ by less than threshold (10%)."""
        # Size compatibility: relative difference
        avg_size = (self.size + other.size) / 2
        if avg_size > 0 and abs(self.size - other.size) / avg_size > threshold:
            return False
        # Color compatibility: relative difference per channel (based on 255 range)
        for attr in ('color_r', 'color_g', 'color_b'):
            a, b = getattr(self, attr), getattr(other, attr)
            if abs(a - b) / 255 > threshold:
                return False
        return True

    def clone(self):
        return Genes(self.size, self.color_r, self.color_g, self.color_b,
                     self.fin_size, self.metabolism)

    @staticmethod
    def crossover(a, b):
        """Mix genes from two parents."""
        child = Genes()
        for attr in ('size', 'color_r', 'color_g', 'color_b', 'fin_size', 'metabolism'):
            if np.random.random() < 0.5:
                setattr(child, attr, getattr(a, attr))
            else:
                setattr(child, attr, getattr(b, attr))
        return child

    @staticmethod
    def mutate(genes, rate=0.15, strength=0.1):
        """Mutate genes in-place."""
        if np.random.random() < rate:
            genes.size = np.clip(genes.size + np.random.normal(0, strength * 0.3), 0.5, 2.0)
        if np.random.random() < rate:
            genes.color_r = np.clip(genes.color_r + np.random.normal(0, 25), 30, 255)
            genes.color_g = np.clip(genes.color_g + np.random.normal(0, 25), 30, 255)
            genes.color_b = np.clip(genes.color_b + np.random.normal(0, 25), 30, 255)
        if np.random.random() < rate:
            genes.fin_size = np.clip(genes.fin_size + np.random.normal(0, strength * 0.2), 0.5, 1.5)
        if np.random.random() < rate:
            genes.metabolism = np.clip(genes.metabolism + np.random.normal(0, strength * 0.2), 0.5, 2.0)

    @staticmethod
    def random():
        """Create random genes."""
        return Genes(
            size=np.random.uniform(0.7, 1.3),
            color_r=np.random.uniform(100, 255),
            color_g=np.random.uniform(60, 200),
            color_b=np.random.uniform(20, 150),
            fin_size=np.random.uniform(0.7, 1.3),
            metabolism=np.random.uniform(0.7, 1.3),
        )

    @staticmethod
    def random_tribe(base=None, count=1, spread=0.05):
        """Create a group of similar genes (a tribe).
        If base is None, generates a random base first."""
        if base is None:
            base = Genes.random()
        tribe = []
        for _ in range(count):
            g = base.clone()
            g.size = np.clip(g.size + np.random.normal(0, spread * 0.3), 0.5, 2.0)
            g.color_r = np.clip(g.color_r + np.random.normal(0, spread * 50), 30, 255)
            g.color_g = np.clip(g.color_g + np.random.normal(0, spread * 50), 30, 255)
            g.color_b = np.clip(g.color_b + np.random.normal(0, spread * 50), 30, 255)
            g.fin_size = np.clip(g.fin_size + np.random.normal(0, spread * 0.2), 0.5, 1.5)
            g.metabolism = np.clip(g.metabolism + np.random.normal(0, spread * 0.2), 0.5, 2.0)
            tribe.append(g)
        return tribe

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    @staticmethod
    def from_dict(d):
        return Genes(**d)
