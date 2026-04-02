"""Spatial hash grid for O(1) neighbor lookups instead of O(N^2).

Divides the world into cells. Each entity is placed in the cell
matching its position. To find neighbors within a radius, only
check the relevant cells.
"""
import config


class SpatialGrid:
    """Fast spatial lookup using a hash grid."""

    def __init__(self, cell_size=None):
        self.cell_size = cell_size or config.VISION_RANGE
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def _key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj):
        """Insert an object with .x and .y attributes."""
        key = self._key(obj.x, obj.y)
        if key in self.cells:
            self.cells[key].append(obj)
        else:
            self.cells[key] = [obj]

    def insert_all(self, objects):
        """Bulk insert a list of objects."""
        self.cells.clear()
        cs = self.cell_size
        for obj in objects:
            key = (int(obj.x // cs), int(obj.y // cs))
            if key in self.cells:
                self.cells[key].append(obj)
            else:
                self.cells[key] = [obj]

    def query(self, x, y, radius):
        """Return all objects within radius of (x, y)."""
        cs = self.cell_size
        r_cells = int(radius // cs) + 1
        cx, cy = int(x // cs), int(y // cs)
        result = []
        r_sq = radius * radius
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell:
                    for obj in cell:
                        ddx = obj.x - x
                        ddy = obj.y - y
                        if ddx * ddx + ddy * ddy <= r_sq:
                            result.append(obj)
        return result

    def query_pairs(self, radius):
        """Return all pairs of objects within radius of each other.
        Each pair returned once as (a, b) where id(a) < id(b)."""
        pairs = []
        r_sq = radius * radius
        visited = set()
        for key, cell in self.cells.items():
            cx, cy = key
            # Check this cell and neighbor cells
            for dx in range(0, 2):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy < 0:
                        continue  # avoid double-checking
                    nkey = (cx + dx, cy + dy)
                    if nkey == key:
                        # Same cell: check all pairs within
                        for i in range(len(cell)):
                            for j in range(i + 1, len(cell)):
                                a, b = cell[i], cell[j]
                                ddx = b.x - a.x
                                ddy = b.y - a.y
                                if ddx * ddx + ddy * ddy <= r_sq:
                                    pairs.append((a, b))
                    else:
                        ncell = self.cells.get(nkey)
                        if ncell:
                            for a in cell:
                                for b in ncell:
                                    ddx = b.x - a.x
                                    ddy = b.y - a.y
                                    if ddx * ddx + ddy * ddy <= r_sq:
                                        pairs.append((a, b))
        return pairs
