"""In-game config menu with adjustable parameters."""
import config


class ConfigItem:
    """One adjustable parameter."""
    __slots__ = ('label', 'attr', 'value', 'min_val', 'max_val', 'step', 'fmt', 'needs_reset')

    def __init__(self, label, attr, min_val, max_val, step, fmt=".1f", needs_reset=False):
        self.label = label
        self.attr = attr
        self.value = getattr(config, attr)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.fmt = fmt
        self.needs_reset = needs_reset

    def increment(self):
        self.value = min(self.max_val, self.value + self.step)
        self._apply()

    def decrement(self):
        self.value = max(self.min_val, self.value - self.step)
        self._apply()

    def _apply(self):
        # Round to avoid float drift
        if isinstance(self.step, int):
            self.value = int(self.value)
        else:
            self.value = round(self.value, 4)
        setattr(config, self.attr, self.value)

    def display_value(self):
        if isinstance(self.value, int):
            return str(self.value)
        return f"{self.value:{self.fmt}}"


class ConfigMenu:
    def __init__(self):
        self.visible = False
        self.selected = 0
        self.pending_reset = False  # whether a reset-requiring param changed

        self.items = [
            # --- Population ---
            ConfigItem("Population size",     "POPULATION_SIZE",    10,  200,  10,  "d",   needs_reset=True),
            ConfigItem("Max population",      "MAX_POPULATION",     20,  300,  10,  "d",   needs_reset=False),
            ConfigItem("Min population",      "MIN_POPULATION",      2,   30,   1,  "d",   needs_reset=False),

            # --- Lifecycle ---
            ConfigItem("Max lifespan (ticks)","MAX_LIFESPAN",     1800,36000, 600, "d",   needs_reset=False),
            ConfigItem("Reproduce food req",  "REPRODUCE_FOOD",      1,   20,   1,  "d",   needs_reset=False),
            ConfigItem("Reproduce energy cost","REPRODUCE_COST",    10,  100,   5, ".0f",  needs_reset=False),
            ConfigItem("Offspring distance",  "OFFSPRING_DISTANCE",  10,  150,  10,  "d",   needs_reset=False),
            ConfigItem("Hall of Fame size",   "HALL_OF_FAME_SIZE",    1,   20,   1,  "d",   needs_reset=True),
            ConfigItem("Mate distance",       "MATE_DISTANCE",       20,  300,  10,  "d",   needs_reset=False),
            ConfigItem("Mate compatibility",  "MATE_COMPATIBILITY", 0.05, 0.50,0.05,".2f",  needs_reset=False),

            # --- Food & Oases ---
            ConfigItem("Food count",          "FOOD_COUNT",          5,  200,   5,  "d",   needs_reset=True),
            ConfigItem("Oasis count",         "OASIS_COUNT",         1,    8,   1,  "d",   needs_reset=True),
            ConfigItem("Oasis width",         "OASIS_RADIUS_W",    100, 1000,  50,  "d",   needs_reset=True),
            ConfigItem("Oasis height",        "OASIS_RADIUS_H",    100,  800,  50,  "d",   needs_reset=True),
            ConfigItem("Eat distance",        "EAT_DISTANCE",        5,   50,   5,  "d",   needs_reset=False),

            # --- Energy ---
            ConfigItem("Initial energy",      "INITIAL_ENERGY",     30,  500,  10,  ".0f", needs_reset=True),
            ConfigItem("Energy decay /tick",  "ENERGY_DECAY",     0.01,  0.5, 0.01, ".2f", needs_reset=False),
            ConfigItem("Food energy",         "FOOD_ENERGY",         5,  100,   5,  ".0f", needs_reset=False),

            # --- Genetics ---
            ConfigItem("Mutation rate",       "MUTATION_RATE",    0.01,  0.5, 0.01, ".2f", needs_reset=False),
            ConfigItem("Mutation strength",   "MUTATION_STRENGTH", 0.05, 1.0, 0.05, ".2f", needs_reset=False),

            # --- Vision ---
            ConfigItem("Vision range",        "VISION_RANGE",       50, 800,   25, ".0f", needs_reset=False),
            ConfigItem("Smell range",         "SMELL_RANGE",       100, 1200,  50, ".0f", needs_reset=False),
            ConfigItem("Smell noise (deg)",   "SMELL_NOISE_DEG",     0, 90,     5, ".0f", needs_reset=False),

            # --- Physics ---
            ConfigItem("Max speed",           "MAX_SPEED",         1.0, 10.0, 0.5, ".1f", needs_reset=False),
            ConfigItem("Tail force",          "TAIL_FORCE",       0.05, 0.5, 0.02, ".2f", needs_reset=False),
            ConfigItem("Side fin force",      "SIDE_FIN_FORCE",   0.01, 0.2, 0.01, ".2f", needs_reset=False),
            ConfigItem("Drag forward",        "DRAG_FORWARD",     0.90, 1.0, 0.01, ".2f", needs_reset=False),
            ConfigItem("Drag lateral",        "DRAG_LATERAL",     0.50, 0.99,0.02, ".2f", needs_reset=False),
            ConfigItem("Weathervane",         "WEATHERVANE",     0.000, 0.01,0.001,".3f", needs_reset=False),
        ]

    def toggle(self):
        self.visible = not self.visible

    def move_up(self):
        self.selected = (self.selected - 1) % len(self.items)

    def move_down(self):
        self.selected = (self.selected + 1) % len(self.items)

    def adjust_right(self):
        item = self.items[self.selected]
        item.increment()
        if item.needs_reset:
            self.pending_reset = True

    def adjust_left(self):
        item = self.items[self.selected]
        item.decrement()
        if item.needs_reset:
            self.pending_reset = True

    def consume_reset(self):
        """Check and clear pending reset flag."""
        if self.pending_reset:
            self.pending_reset = False
            return True
        return False
