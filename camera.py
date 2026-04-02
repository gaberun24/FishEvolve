"""Camera with zoom and pan for large worlds."""
import config


class Camera:
    def __init__(self):
        self.x = config.WORLD_WIDTH / 2   # center of view in world coords
        self.y = config.WORLD_HEIGHT / 2
        self.zoom = config.SCREEN_WIDTH / config.WORLD_WIDTH  # fit whole world
        self.min_zoom = 0.05
        self.max_zoom = 2.0
        self.target_zoom = self.zoom
        self.pan_speed = 10  # pixels per frame at current zoom

    def world_to_screen(self, wx, wy):
        """Convert world coords to screen coords."""
        sx = (wx - self.x) * self.zoom + config.SCREEN_WIDTH / 2
        sy = (wy - self.y) * self.zoom + config.SCREEN_HEIGHT / 2
        return sx, sy

    def screen_to_world(self, sx, sy):
        """Convert screen coords to world coords."""
        wx = (sx - config.SCREEN_WIDTH / 2) / self.zoom + self.x
        wy = (sy - config.SCREEN_HEIGHT / 2) / self.zoom + self.y
        return wx, wy

    def visible_rect(self):
        """Return (left, top, right, bottom) in world coords."""
        half_w = config.SCREEN_WIDTH / 2 / self.zoom
        half_h = config.SCREEN_HEIGHT / 2 / self.zoom
        return (self.x - half_w, self.y - half_h,
                self.x + half_w, self.y + half_h)

    def is_visible(self, wx, wy, margin=50):
        """Check if a world point is visible on screen."""
        l, t, r, b = self.visible_rect()
        return (l - margin < wx < r + margin and
                t - margin < wy < b + margin)

    def zoom_at(self, screen_x, screen_y, factor):
        """Zoom toward a screen point."""
        # World point under cursor before zoom
        wx, wy = self.screen_to_world(screen_x, screen_y)
        self.zoom *= factor
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
        # Adjust center so the world point stays under cursor
        self.x = wx - (screen_x - config.SCREEN_WIDTH / 2) / self.zoom
        self.y = wy - (screen_y - config.SCREEN_HEIGHT / 2) / self.zoom

    def pan(self, dx, dy):
        """Pan by screen pixels."""
        self.x -= dx / self.zoom
        self.y -= dy / self.zoom

    def follow(self, wx, wy, smoothing=0.05):
        """Smoothly follow a world point."""
        self.x += (wx - self.x) * smoothing
        self.y += (wy - self.y) * smoothing

    def fit_world(self):
        """Zoom to fit the whole world."""
        zx = config.SCREEN_WIDTH / config.WORLD_WIDTH
        zy = config.SCREEN_HEIGHT / config.WORLD_HEIGHT
        self.zoom = min(zx, zy)
        self.x = config.WORLD_WIDTH / 2
        self.y = config.WORLD_HEIGHT / 2
