# Neuroevolution Fish Simulation - Configuration
import math

# World (larger world for ecosystem)
WORLD_WIDTH = 6000
WORLD_HEIGHT = 4000
WALL_BOUNCE = 0.3

# Screen / camera
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Food oases: list of (center_x_frac, center_y_frac, w, h) as fractions of world size
OASIS_COUNT = 4
OASIS_RADIUS_W = 350   # half-width of each oasis
OASIS_RADIUS_H = 250   # half-height of each oasis

# Population
POPULATION_SIZE = 50
GENERATION_TIME_SEC = 15
ELITISM_COUNT = 2

# Vision
VISION_RAYS = 12           # number of vision sectors (was 8)
VISION_RANGE = 300.0       # max vision distance in pixels
FRONT_BLIND_DEG = 5.0      # front blind spot (total degrees)
BACK_BLIND_DEG = 10.0      # back blind spot (total degrees)
VISION_CONE_HALF = None     # computed below

# Lateral line (360° pressure/motion sensing)
LATERAL_LINE_RANGE = 200.0  # shorter than vision but works in blind spots
LATERAL_LINE_SECTORS = 4    # front, right, back, left

# Smell
SMELL_NOISE_DEG = 30.0     # smell only gives rough direction (±30° noise)
SMELL_RANGE = 600.0        # smell works further than vision but imprecise

# Addressed memory matrix: NxN grid, accessed via read/write heads
MEMORY_SIZE = 4           # 4x4 = 16 memory cells

# Neural network topology: [inputs, hidden1, hidden2, outputs]
# --- SENSORY INPUTS (72) ---
# Vision:       12 rays × 4 ch (food, fish_dist, fish_kin, wall) = 48
# Lateral line: 4 sectors × 2 ch (pressure, motion) = 8
# Smell:        2 (rough angle, rough distance)
# Self:         3 (speed, angular_vel, energy_frac)
# Mating:       1 (nearby mating signal)
# Position:     2 (normalized x, y in world)
# Compass:      2 (sin/cos of heading)
# Oasis homing: 2 (angle + distance to nearest oasis)
# Tribe sense:  2 (nearby kin count, avg direction to kin)
# Memory read:  2 (values from 2 read heads)
# --- RECURRENT (48) ---
# Hidden1 state fed back from previous tick
# --- TOTAL INPUT = 72 + 48 = 120 ---
# --- OUTPUTS (12) ---
# Motor:        5 (left_fin, right_fin, tail, mouth, mating_signal)
# Write head:   3 (write_row, write_col, write_value)
# Read heads:   4 (read1_row, read1_col, read2_row, read2_col)
SENSORY_INPUTS = 72
RECURRENT_SIZE = 48           # hidden1 feeds back as extra input
NN_TOPOLOGY = [SENSORY_INPUTS + RECURRENT_SIZE, 48, 24, 12]

# Genetics
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.3
CROSSOVER_RATE = 0.7

# Fish physics
MAX_SPEED = 3.0
DRAG = 0.97
ANGULAR_DRAG = 0.85
DRAG_FORWARD = 0.97       # low drag along body axis (streamlined)
DRAG_LATERAL = 0.80       # high drag sideways (flat side hits water)
WEATHERVANE = 0.002       # passive torque aligning body to velocity
SIDE_FIN_FORCE = 0.06
SIDE_FIN_TORQUE = 0.025
TAIL_FORCE = 0.18
MAX_ANGULAR_VEL = 0.06
MOUTH_OFFSET = 0.85
FISH_BODY_RADIUS = 25.0    # collision radius for fish body
FISH_PUSH_FORCE = 0.5      # how hard fish push each other apart

# Energy
INITIAL_ENERGY = 100.0
ENERGY_DECAY = 0.05
FOOD_ENERGY = 30.0

# Continuous ecosystem
MAX_LIFESPAN = 7200       # 2 minutes at 60fps
REPRODUCE_FOOD = 5        # food eaten needed to reproduce
REPRODUCE_COST = 40.0     # energy cost to reproduce
MAX_POPULATION = 120      # cap - no reproduction above this
MIN_POPULATION = 5        # hall of fame respawn if below this
OFFSPRING_DISTANCE = 40   # spawn distance from parent
MATE_DISTANCE = 80        # max distance to find a mate
MATE_COMPATIBILITY = 0.10 # max 10% gene difference to mate
HALL_OF_FAME_SIZE = 5     # best fish ever remembered

# Food (distributed across oases)
FOOD_COUNT = 40
FOOD_RADIUS = 6
EAT_DISTANCE = 15
FOOD_MARGIN = 40
FOOD_DRAG = 0.90           # food slows down quickly when pushed
FOOD_PUSH_FORCE = 1.5      # how hard fish push food

# Rendering
FPS = 60
FISH_SCALE = 0.18

# Colors
BG_COLOR = (20, 30, 50)
WATER_COLOR = (25, 40, 65)
FOOD_COLOR = (100, 220, 120)
FOOD_GLOW_COLOR = (80, 200, 100)
HUD_COLOR = (200, 210, 220)
HUD_BG_COLOR = (15, 20, 35, 180)
GRID_COLOR = (30, 45, 70)

# --- Computed vision constants ---
_front_half = math.radians(FRONT_BLIND_DEG / 2)
_back_half = math.radians(BACK_BLIND_DEG / 2)
_visible_arc = 2 * math.pi - math.radians(FRONT_BLIND_DEG) - math.radians(BACK_BLIND_DEG)
VISION_CONE_HALF = _visible_arc / VISION_RAYS / 2

# Ray angles relative to heading (0 = forward)
# Spread evenly across the visible arc, avoiding blind spots
VISION_RAY_ANGLES = []
_sector_size = _visible_arc / VISION_RAYS
for i in range(VISION_RAYS):
    # Start from right side of front blind spot, go clockwise
    _angle = _front_half + _sector_size * (i + 0.5)
    # Wrap: if we cross the back blind spot, skip over it
    if _angle > math.pi - _back_half:
        _angle += 2 * _back_half
    # Normalize to [-pi, pi]
    if _angle > math.pi:
        _angle -= 2 * math.pi
    VISION_RAY_ANGLES.append(_angle)
