import math
import numpy as np
import config
from utils import distance


def get_oases():
    """Return list of (cx, cy) oasis center positions, evenly spread."""
    n = config.OASIS_COUNT
    w, h = config.WORLD_WIDTH, config.WORLD_HEIGHT
    # Arrange oases in a grid-like pattern with margins
    margin_x = w * 0.2
    margin_y = h * 0.2
    cols = max(1, round(math.sqrt(n * w / h)))
    rows = max(1, math.ceil(n / cols))
    oases = []
    for i in range(n):
        r = i // cols
        c = i % cols
        cx = margin_x + (w - 2 * margin_x) * (c + 0.5) / cols
        cy = margin_y + (h - 2 * margin_y) * (r + 0.5) / rows
        oases.append((cx, cy))
    return oases


class FoodItem:
    __slots__ = ('x', 'y', 'vx', 'vy')

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        """Move food (when pushed)."""
        self.vx *= config.FOOD_DRAG
        self.vy *= config.FOOD_DRAG
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        m = config.FOOD_MARGIN
        if self.x < m:
            self.x = m
            self.vx = abs(self.vx) * 0.3
        elif self.x > config.WORLD_WIDTH - m:
            self.x = config.WORLD_WIDTH - m
            self.vx = -abs(self.vx) * 0.3
        if self.y < m:
            self.y = m
            self.vy = abs(self.vy) * 0.3
        elif self.y > config.WORLD_HEIGHT - m:
            self.y = config.WORLD_HEIGHT - m
            self.vy = -abs(self.vy) * 0.3


class FoodManager:
    def __init__(self):
        self.items = []

    def init_food(self):
        self.items.clear()
        for _ in range(config.FOOD_COUNT):
            self.spawn_one()

    def spawn_one(self):
        """Spawn food in a random oasis."""
        oases = get_oases()
        cx, cy = oases[np.random.randint(len(oases))]
        rw = config.OASIS_RADIUS_W
        rh = config.OASIS_RADIUS_H
        x = np.random.uniform(cx - rw, cx + rw)
        y = np.random.uniform(cy - rh, cy + rh)
        x = np.clip(x, config.FOOD_MARGIN, config.WORLD_WIDTH - config.FOOD_MARGIN)
        y = np.clip(y, config.FOOD_MARGIN, config.WORLD_HEIGHT - config.FOOD_MARGIN)
        self.items.append(FoodItem(x, y))

    def update(self):
        for item in self.items:
            item.update()

    def push_food_from_fish(self, fishes):
        for fish in fishes:
            if not fish.alive:
                continue
            for item in self.items:
                d = distance(fish.x, fish.y, item.x, item.y)
                push_dist = config.FISH_BODY_RADIUS + config.FOOD_RADIUS
                if d < push_dist and d > 0.1:
                    dx = (item.x - fish.x) / d
                    dy = (item.y - fish.y) / d
                    overlap = push_dist - d
                    item.vx += dx * overlap * config.FOOD_PUSH_FORCE * 0.1
                    item.vy += dy * overlap * config.FOOD_PUSH_FORCE * 0.1
                    item.x += dx * overlap * 0.5
                    item.y += dy * overlap * 0.5

    def find_nearest(self, x, y):
        if not self.items:
            return config.WORLD_WIDTH / 2, config.WORLD_HEIGHT / 2, 1e9
        best_dist = float('inf')
        best = self.items[0]
        for item in self.items:
            d = distance(x, y, item.x, item.y)
            if d < best_dist:
                best_dist = d
                best = item
        return best.x, best.y, best_dist

    def check_eating(self, fishes):
        eaten_count = 0
        for fish in fishes:
            if not fish.alive or fish.mouth_open < 0.5:
                continue
            mx, my = fish.mouth_x, fish.mouth_y
            to_remove = []
            for i, item in enumerate(self.items):
                if distance(mx, my, item.x, item.y) < config.EAT_DISTANCE:
                    fish.eat()
                    to_remove.append(i)
                    eaten_count += 1
                    break
            for i in reversed(to_remove):
                self.items.pop(i)
                self.spawn_one()
        return eaten_count

    def reset(self):
        self.init_food()
