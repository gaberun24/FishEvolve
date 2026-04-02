import math
import numpy as np
import config
from config import (
    WALL_BOUNCE,
    MOUTH_OFFSET, FISH_SCALE,
    VISION_RAYS, VISION_RANGE, VISION_RAY_ANGLES, VISION_CONE_HALF,
    SMELL_NOISE_DEG, SMELL_RANGE, NN_TOPOLOGY,
)
from neural_network import NeuralNetwork
from genes import Genes
from utils import angle_diff, angle_to, distance, clamp


class Fish:
    BASE_BODY_HALF_LEN = 140 * FISH_SCALE

    def __init__(self, x=None, y=None, angle=None, genes=None):
        self.x = x if x is not None else np.random.uniform(100, config.WORLD_WIDTH - 100)
        self.y = y if y is not None else np.random.uniform(100, config.WORLD_HEIGHT - 100)
        self.angle = angle if angle is not None else np.random.uniform(0, 2 * math.pi)
        self.vx = 0.0
        self.vy = 0.0
        self.angular_vel = 0.0
        self.speed = 0.0

        # Genes
        self.genes = genes if genes is not None else Genes()

        self.brain = NeuralNetwork(NN_TOPOLOGY, recurrent_size=config.RECURRENT_SIZE)
        self.food_eaten = 0
        self.total_food = 0  # lifetime total (for hall of fame)
        self.distance_traveled = 0.0
        self.energy = config.INITIAL_ENERGY
        self.alive = True
        self.age = 0
        self.children = 0  # how many offspring produced

        # Animation state
        self.fin_phase = 0.0
        self.body_wave_phase = 0.0
        self.mouth_open = 0.0
        self.left_fin_force = 0.0
        self.right_fin_force = 0.0
        self.tail_force = 0.0
        self.mating_signal = 0.0  # 0-1, broadcast to nearby fish

        # Addressed memory matrix (4x4)
        ms = config.MEMORY_SIZE
        self.memory_matrix = [[0.0] * ms for _ in range(ms)]
        # Read head results (from previous tick)
        self.mem_read1 = 0.0
        self.mem_read2 = 0.0
        # Last write/read addresses (for visualization)
        self.mem_write_addr = (0, 0)
        self.mem_read1_addr = (0, 0)
        self.mem_read2_addr = (0, 0)

        # Vision results (for debug rendering)
        self.vision_food = [0.0] * VISION_RAYS
        self.vision_fish = [0.0] * VISION_RAYS
        self.vision_fish_kin = [0.0] * VISION_RAYS
        self.vision_wall = [0.0] * VISION_RAYS

        # Sense results (for dashboard rendering)
        self.lateral_pressure = [0.0] * 4   # front, right, back, left
        self.lateral_motion = [0.0] * 4
        self.smell_angle = 0.0              # relative angle to food [-1,1]
        self.smell_dist = 1.0               # 0=close, 1=far/none
        self.oasis_angle = 0.0              # relative angle to nearest oasis
        self.oasis_dist = 1.0
        self.tribe_count = 0.0              # nearby kin (0-1)
        self.tribe_dir = 0.0                # direction to kin

    # --- Gene-derived properties ---
    @property
    def body_half_len(self):
        return self.BASE_BODY_HALF_LEN * self.genes.size

    @property
    def body_radius(self):
        return config.FISH_BODY_RADIUS * self.genes.size

    @property
    def max_speed(self):
        # Smaller = faster, bigger = slower
        return config.MAX_SPEED * (1.3 - 0.3 * self.genes.size) * (0.7 + 0.3 * self.genes.metabolism)

    @property
    def energy_decay(self):
        # Bigger + faster metabolism = more energy consumption
        return config.ENERGY_DECAY * self.genes.size * self.genes.metabolism

    @property
    def push_force(self):
        # Bigger fish push harder
        return config.FISH_PUSH_FORCE * self.genes.size * self.genes.size

    @property
    def tail_power(self):
        return config.TAIL_FORCE * (0.7 + 0.3 * self.genes.size) * self.genes.metabolism

    @property
    def side_fin_power(self):
        return config.SIDE_FIN_FORCE * self.genes.fin_size

    @property
    def side_fin_torque(self):
        return config.SIDE_FIN_TORQUE * self.genes.fin_size

    @property
    def color(self):
        return self.genes.color

    @property
    def mouth_x(self):
        return self.x + math.cos(self.angle) * self.body_half_len * MOUTH_OFFSET

    @property
    def mouth_y(self):
        return self.y + math.sin(self.angle) * self.body_half_len * MOUTH_OFFSET

    @property
    def can_reproduce(self):
        return (self.food_eaten >= config.REPRODUCE_FOOD and
                self.energy > config.REPRODUCE_COST and
                self.alive)

    # --- Vision (12 rays × 4 channels) ---
    def see(self, food_items, all_fish):
        for r in range(VISION_RAYS):
            ray_angle = self.angle + VISION_RAY_ANGLES[r]
            best_food = 1.0
            best_fish = 1.0
            best_fish_kin = 0.0   # 0 = stranger/none, 1 = compatible kin
            best_wall = 1.0

            for item in food_items:
                d = distance(self.x, self.y, item.x, item.y)
                if d > VISION_RANGE or d < 0.1:
                    continue
                obj_angle = math.atan2(item.y - self.y, item.x - self.x)
                diff = angle_diff(ray_angle, obj_angle)
                if abs(diff) < VISION_CONE_HALF:
                    proximity = d / VISION_RANGE
                    if proximity < best_food:
                        best_food = proximity

            best_fish_d = VISION_RANGE
            best_fish_ref = None
            for other in all_fish:
                if other is self or not other.alive:
                    continue
                d = distance(self.x, self.y, other.x, other.y)
                if d > VISION_RANGE or d < 0.1:
                    continue
                obj_angle = math.atan2(other.y - self.y, other.x - self.x)
                diff = angle_diff(ray_angle, obj_angle)
                if abs(diff) < VISION_CONE_HALF:
                    if d < best_fish_d:
                        best_fish_d = d
                        best_fish_ref = other
            if best_fish_ref is not None:
                best_fish = best_fish_d / VISION_RANGE
                # Kin recognition: 1 = compatible, 0 = stranger
                best_fish_kin = 1.0 if self.genes.compatible_with(
                    best_fish_ref.genes, config.MATE_COMPATIBILITY) else 0.0

            # Wall distance
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            t_min = VISION_RANGE
            if dx > 1e-9:
                t_min = min(t_min, (config.WORLD_WIDTH - self.x) / dx)
            elif dx < -1e-9:
                t_min = min(t_min, -self.x / dx)
            if dy > 1e-9:
                t_min = min(t_min, (config.WORLD_HEIGHT - self.y) / dy)
            elif dy < -1e-9:
                t_min = min(t_min, -self.y / dy)
            t_min = max(0, t_min)
            best_wall = clamp(t_min / VISION_RANGE, 0, 1)

            self.vision_food[r] = best_food
            self.vision_fish[r] = best_fish
            self.vision_fish_kin[r] = best_fish_kin
            self.vision_wall[r] = best_wall

    # --- Lateral line (360° pressure/motion sense) ---
    def sense_lateral_line(self, all_fish, food_items):
        """Sense nearby objects in 4 quadrants (front/right/back/left).
        Returns pressure (how many things nearby) and motion (speed) per sector."""
        ll_range = config.LATERAL_LINE_RANGE
        # 4 sectors: 0=front, 1=right, 2=back, 3=left
        pressure = [0.0] * 4
        motion = [0.0] * 4

        for other in all_fish:
            if other is self or not other.alive:
                continue
            d = distance(self.x, self.y, other.x, other.y)
            if d > ll_range or d < 0.1:
                continue
            # Which sector?
            obj_angle = math.atan2(other.y - self.y, other.x - self.x)
            rel = angle_diff(obj_angle, self.angle)  # relative to heading
            if abs(rel) < math.pi / 4:
                sector = 0  # front
            elif rel >= math.pi / 4 and rel < 3 * math.pi / 4:
                sector = 1  # right
            elif rel <= -math.pi / 4 and rel > -3 * math.pi / 4:
                sector = 3  # left
            else:
                sector = 2  # back
            strength = 1.0 - d / ll_range
            pressure[sector] += strength
            # Motion = other's speed projected toward us
            other_speed = math.sqrt(other.vx**2 + other.vy**2)
            motion[sector] = max(motion[sector], other_speed / self.max_speed * strength)

        # Also sense food
        for item in food_items:
            d = distance(self.x, self.y, item.x, item.y)
            if d > ll_range or d < 0.1:
                continue
            obj_angle = math.atan2(item.y - self.y, item.x - self.x)
            rel = angle_diff(obj_angle, self.angle)
            if abs(rel) < math.pi / 4:
                sector = 0
            elif rel >= math.pi / 4 and rel < 3 * math.pi / 4:
                sector = 1
            elif rel <= -math.pi / 4 and rel > -3 * math.pi / 4:
                sector = 3
            else:
                sector = 2
            strength = (1.0 - d / ll_range) * 0.3  # food = weaker signal
            pressure[sector] += strength

        # Clamp
        for i in range(4):
            pressure[i] = clamp(pressure[i], 0, 1)
            motion[i] = clamp(motion[i], 0, 1)
        return pressure, motion

    def smell(self, food_items):
        best_dist = float('inf')
        best_item = None
        for item in food_items:
            d = distance(self.x, self.y, item.x, item.y)
            if d < best_dist:
                best_dist = d
                best_item = item

        if best_item is None or best_dist > SMELL_RANGE:
            return 0.0, 1.0

        true_angle = angle_to(self.x, self.y, best_item.x, best_item.y)
        noise = np.random.uniform(
            -math.radians(SMELL_NOISE_DEG),
            math.radians(SMELL_NOISE_DEG)
        )
        noisy_rel = angle_diff(self.angle, true_angle + noise) / math.pi
        rough_dist = clamp(best_dist / SMELL_RANGE, 0, 1)
        return noisy_rel, rough_dist

    def sense_mating_signal(self, all_fish):
        """Sense strongest mating signal from nearby compatible fish."""
        best = 0.0
        for other in all_fish:
            if other is self or not other.alive:
                continue
            d = distance(self.x, self.y, other.x, other.y)
            if d < VISION_RANGE and other.mating_signal > 0.1:
                strength = other.mating_signal * (1.0 - d / VISION_RANGE)
                if strength > best:
                    best = strength
        return best

    def sense_oasis(self):
        """Direction and distance to nearest oasis."""
        from food import get_oases
        oases = get_oases()
        if not oases:
            return 0.0, 1.0
        best_d = float('inf')
        best_ox, best_oy = oases[0]
        for ox, oy in oases:
            d = distance(self.x, self.y, ox, oy)
            if d < best_d:
                best_d = d
                best_ox, best_oy = ox, oy
        # Relative angle to oasis (normalized to [-1, 1])
        oasis_angle = angle_to(self.x, self.y, best_ox, best_oy)
        rel_angle = angle_diff(oasis_angle, self.angle)
        # Distance (normalized, 0=close, 1=far)
        max_d = math.sqrt(config.WORLD_WIDTH**2 + config.WORLD_HEIGHT**2)
        return clamp(rel_angle / math.pi, -1, 1), clamp(best_d / max_d, 0, 1)

    def sense_tribe(self, all_fish):
        """Sense nearby compatible fish: count and average direction."""
        tribe_range = VISION_RANGE * 2  # can sense kin further than vision
        count = 0
        sum_dx, sum_dy = 0.0, 0.0
        for other in all_fish:
            if other is self or not other.alive:
                continue
            d = distance(self.x, self.y, other.x, other.y)
            if d < tribe_range and d > 0.1:
                if self.genes.compatible_with(other.genes, config.MATE_COMPATIBILITY):
                    count += 1
                    sum_dx += (other.x - self.x) / d
                    sum_dy += (other.y - self.y) / d
        # Normalize count (0-1 range, capped at ~10 nearby kin)
        count_norm = clamp(count / 10.0, 0, 1)
        # Average direction as relative angle
        if count > 0:
            tribe_angle = math.atan2(sum_dy / count, sum_dx / count)
            rel_angle = angle_diff(tribe_angle, self.angle)
            dir_norm = clamp(rel_angle / math.pi, -1, 1)
        else:
            dir_norm = 0.0
        return count_norm, dir_norm

    def get_inputs(self, food_items, all_fish):
        self.see(food_items, all_fish)
        pressure, motion = self.sense_lateral_line(all_fish, food_items)
        smell_angle, smell_dist = self.smell(food_items)
        oasis_angle, oasis_dist = self.sense_oasis()
        tribe_count, tribe_dir = self.sense_tribe(all_fish)

        # Store for dashboard visualization
        self.lateral_pressure = pressure
        self.lateral_motion = motion
        self.smell_angle = smell_angle
        self.smell_dist = smell_dist
        self.oasis_angle = oasis_angle
        self.oasis_dist = oasis_dist
        self.tribe_count = tribe_count
        self.tribe_dir = tribe_dir

        inputs = []
        # Vision: 12 rays × 4 channels = 48
        for r in range(VISION_RAYS):
            inputs.append(self.vision_food[r])
            inputs.append(self.vision_fish[r])
            inputs.append(self.vision_fish_kin[r])
            inputs.append(self.vision_wall[r])
        # Lateral line: 4 sectors × 2 channels = 8
        for s in range(4):
            inputs.append(pressure[s])
            inputs.append(motion[s])
        # Smell (2)
        inputs.append(smell_angle)
        inputs.append(smell_dist)
        # Self (3)
        inputs.append(clamp(self.speed / self.max_speed, 0, 1))
        inputs.append(clamp(self.angular_vel / config.MAX_ANGULAR_VEL, -1, 1))
        inputs.append(clamp(self.energy / config.INITIAL_ENERGY, 0, 1))
        # Mating (1)
        inputs.append(self.sense_mating_signal(all_fish))
        # Position (2) - "where am I in the world"
        inputs.append(self.x / config.WORLD_WIDTH)
        inputs.append(self.y / config.WORLD_HEIGHT)
        # Compass (2) - "which direction am I facing"
        inputs.append(math.sin(self.angle))
        inputs.append(math.cos(self.angle))
        # Oasis homing (2) - "where is food home"
        inputs.append(oasis_angle)
        inputs.append(oasis_dist)
        # Tribe sense (2) - "where are my kind"
        inputs.append(tribe_count)
        inputs.append(tribe_dir)
        # Memory read heads (2) - values read from last tick's addresses
        inputs.append(self.mem_read1)
        inputs.append(self.mem_read2)

        return inputs

    # --- Physics update ---
    def update(self, food_items, all_fish):
        if not self.alive:
            return

        inputs = self.get_inputs(food_items, all_fish)
        outputs = self.brain.feed_forward(inputs)

        left_fin = float(outputs[0])
        right_fin = float(outputs[1])
        tail = float(outputs[2])
        mouth = float(outputs[3])
        self.mating_signal = clamp((float(outputs[4]) + 1) / 2, 0, 1)  # tanh -> 0-1

        # Addressed memory: write head (3 outputs) + 2 read heads (4 outputs)
        ms = config.MEMORY_SIZE
        # Write: convert tanh outputs to row/col indices
        wr = int(clamp((float(outputs[5]) + 1) / 2 * ms, 0, ms - 1))
        wc = int(clamp((float(outputs[6]) + 1) / 2 * ms, 0, ms - 1))
        wv = float(outputs[7])  # value to write (tanh, -1 to 1)
        self.memory_matrix[wr][wc] = wv
        self.mem_write_addr = (wr, wc)
        # Read head 1
        r1r = int(clamp((float(outputs[8]) + 1) / 2 * ms, 0, ms - 1))
        r1c = int(clamp((float(outputs[9]) + 1) / 2 * ms, 0, ms - 1))
        self.mem_read1 = self.memory_matrix[r1r][r1c]
        self.mem_read1_addr = (r1r, r1c)
        # Read head 2
        r2r = int(clamp((float(outputs[10]) + 1) / 2 * ms, 0, ms - 1))
        r2c = int(clamp((float(outputs[11]) + 1) / 2 * ms, 0, ms - 1))
        self.mem_read2 = self.memory_matrix[r2r][r2c]
        self.mem_read2_addr = (r2r, r2c)

        self.left_fin_force = left_fin
        self.right_fin_force = right_fin
        self.tail_force = max(0, tail)

        # Side fins (gene-scaled)
        side_thrust = (left_fin + right_fin) * self.side_fin_power
        self.vx += math.cos(self.angle) * side_thrust
        self.vy += math.sin(self.angle) * side_thrust
        self.angular_vel += (left_fin - right_fin) * self.side_fin_torque

        # Tail (gene-scaled)
        tail_thrust = self.tail_force * self.tail_power
        self.vx += math.cos(self.angle) * tail_thrust
        self.vy += math.sin(self.angle) * tail_thrust

        # Anisotropic drag
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        v_forward = self.vx * cos_a + self.vy * sin_a
        v_lateral = -self.vx * sin_a + self.vy * cos_a

        # Bigger fins = more lateral drag (better at not drifting sideways)
        lat_drag = config.DRAG_LATERAL * (1.0 - 0.1 * (self.genes.fin_size - 1.0))
        v_forward *= config.DRAG_FORWARD
        v_lateral *= lat_drag

        self.vx = v_forward * cos_a - v_lateral * sin_a
        self.vy = v_forward * sin_a + v_lateral * cos_a

        # Weathervane
        self.speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if self.speed > 0.1:
            vel_angle = math.atan2(self.vy, self.vx)
            misalignment = angle_diff(self.angle, vel_angle)
            self.angular_vel += misalignment * self.speed * config.WEATHERVANE

        self.angular_vel *= config.ANGULAR_DRAG
        self.angular_vel = clamp(self.angular_vel, -config.MAX_ANGULAR_VEL, config.MAX_ANGULAR_VEL)

        # Speed cap (gene-scaled)
        ms = self.max_speed
        if self.speed > ms:
            scale = ms / self.speed
            self.vx *= scale
            self.vy *= scale
            self.speed = ms

        # Integrate
        old_x, old_y = self.x, self.y
        self.x += self.vx
        self.y += self.vy
        self.angle += self.angular_vel
        self.angle %= 2 * math.pi

        self.distance_traveled += distance(old_x, old_y, self.x, self.y)

        # Wall collision
        margin = 10
        if self.x < margin:
            self.x = margin
            self.vx = abs(self.vx) * WALL_BOUNCE
        elif self.x > config.WORLD_WIDTH - margin:
            self.x = config.WORLD_WIDTH - margin
            self.vx = -abs(self.vx) * WALL_BOUNCE
        if self.y < margin:
            self.y = margin
            self.vy = abs(self.vy) * WALL_BOUNCE
        elif self.y > config.WORLD_HEIGHT - margin:
            self.y = config.WORLD_HEIGHT - margin
            self.vy = -abs(self.vy) * WALL_BOUNCE

        # Animations
        target_mouth = 1.0 if mouth > 0 else 0.0
        self.mouth_open += (target_mouth - self.mouth_open) * 0.2
        fin_activity = max(abs(left_fin), abs(right_fin), self.tail_force)
        self.fin_phase += 0.15 + fin_activity * 0.2
        self.body_wave_phase += 0.08 + self.tail_force * 0.2

        # Energy (gene-scaled decay)
        self.energy -= self.energy_decay
        if self.energy <= 0:
            self.energy = 0
            self.alive = False

        # Max lifespan
        if self.age >= config.MAX_LIFESPAN:
            self.alive = False

        self.age += 1

    def eat(self):
        self.food_eaten += 1
        self.total_food += 1
        self.energy = min(config.INITIAL_ENERGY * 1.5, self.energy + config.FOOD_ENERGY)

    def find_mate(self, population):
        """Find a compatible nearby mate."""
        best = None
        best_dist = config.MATE_DISTANCE
        for other in population:
            if other is self or not other.can_reproduce:
                continue
            d = distance(self.x, self.y, other.x, other.y)
            if d < best_dist and self.genes.compatible_with(other.genes, config.MATE_COMPATIBILITY):
                best = other
                best_dist = d
        return best

    def reproduce(self, mate):
        """Create offspring from two parents. Returns new Fish or None."""
        if not self.can_reproduce or not mate.can_reproduce:
            return None

        # Both parents pay
        self.energy -= config.REPRODUCE_COST
        self.food_eaten = 0
        self.children += 1
        mate.energy -= config.REPRODUCE_COST
        mate.food_eaten = 0
        mate.children += 1

        # Offspring position: between parents
        off_angle = np.random.uniform(0, 2 * math.pi)
        mx = (self.x + mate.x) / 2
        my = (self.y + mate.y) / 2
        ox = mx + math.cos(off_angle) * config.OFFSPRING_DISTANCE * 0.5
        oy = my + math.sin(off_angle) * config.OFFSPRING_DISTANCE * 0.5
        ox = clamp(ox, 20, config.WORLD_WIDTH - 20)
        oy = clamp(oy, 20, config.WORLD_HEIGHT - 20)

        # Crossover brain + genes from both parents, then mutate
        child_genes = Genes.crossover(self.genes, mate.genes)
        child = Fish(x=ox, y=oy, genes=child_genes)
        child.brain = NeuralNetwork.crossover(self.brain, mate.brain)
        NeuralNetwork.mutate(child.brain, config.MUTATION_RATE, config.MUTATION_STRENGTH)
        child.brain.reset_state()  # fresh mind for newborn
        Genes.mutate(child.genes)

        return child


def push_fish_apart(population):
    """Push overlapping fish apart, accounting for different sizes."""
    for i, a in enumerate(population):
        if not a.alive:
            continue
        for j in range(i + 1, len(population)):
            b = population[j]
            if not b.alive:
                continue
            d = distance(a.x, a.y, b.x, b.y)
            min_dist = a.body_radius + b.body_radius
            if d < min_dist and d > 0.1:
                dx = (b.x - a.x) / d
                dy = (b.y - a.y) / d
                overlap = min_dist - d
                # Heavier fish pushes lighter fish more
                total_mass = a.genes.size + b.genes.size
                a_ratio = b.genes.size / total_mass  # lighter fish gets pushed more
                b_ratio = a.genes.size / total_mass
                force = overlap * 0.3
                a.vx -= dx * force * a_ratio
                a.vy -= dy * force * a_ratio
                b.vx += dx * force * b_ratio
                b.vy += dy * force * b_ratio
                a.x -= dx * overlap * a_ratio * 0.5
                a.y -= dy * overlap * a_ratio * 0.5
                b.x += dx * overlap * b_ratio * 0.5
                b.y += dy * overlap * b_ratio * 0.5
