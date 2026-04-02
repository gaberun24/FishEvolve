import math
import numpy as np


def angle_diff(a, b):
    """Signed shortest-arc angle difference from a to b, in [-pi, pi]."""
    d = (b - a) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def angle_to(x1, y1, x2, y2):
    """Angle from (x1,y1) to (x2,y2)."""
    return math.atan2(y2 - y1, x2 - x1)


def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def rand_range(lo, hi):
    return np.random.uniform(lo, hi)


def rand_gaussian(mean=0.0, std=1.0):
    return np.random.normal(mean, std)


def wall_distances(x, y, angle, w, h):
    """Returns normalized distances [ahead, left, right, behind] to walls.
    Distances are normalized by the world diagonal."""
    diag = math.sqrt(w * w + h * h)
    dirs = [
        angle,                    # ahead
        angle + math.pi / 2,     # left
        angle - math.pi / 2,     # right
        angle + math.pi,         # behind
    ]
    dists = []
    for d in dirs:
        dx = math.cos(d)
        dy = math.sin(d)
        # find distance to wall in this direction via ray-box intersection
        t_min = float('inf')
        if dx > 1e-9:
            t_min = min(t_min, (w - x) / dx)
        elif dx < -1e-9:
            t_min = min(t_min, -x / dx)
        if dy > 1e-9:
            t_min = min(t_min, (h - y) / dy)
        elif dy < -1e-9:
            t_min = min(t_min, -y / dy)
        t_min = max(0, t_min)
        dists.append(clamp(t_min / diag, 0, 1))
    return dists
