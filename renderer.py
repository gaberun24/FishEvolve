import math
import pygame
import config
from config import (
    FISH_SCALE, FPS,
    BG_COLOR, WATER_COLOR, FOOD_COLOR, FOOD_GLOW_COLOR,
    HUD_COLOR, GRID_COLOR, FOOD_RADIUS,
    VISION_RAYS, VISION_RAY_ANGLES, VISION_RANGE,
)
from food import get_oases
from utils import clamp


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont('Consolas', 16)
        self.font_big = pygame.font.SysFont('Consolas', 22, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 12)

    def draw(self, world, camera, speed_mult, paused, message=None):
        self.screen.fill(BG_COLOR)
        self._draw_grid(camera)
        self._draw_oases(camera)
        self._draw_food(world.food.items, camera)
        self._draw_fishes(world.population, world.selected_fish, camera)
        self._draw_hud(world, speed_mult, paused)
        self._draw_pop_chart(world)
        self._draw_minimap(world, camera)
        if message:
            self._draw_message(message)

    def _draw_grid(self, camera):
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        spacing = 200
        # Only draw visible grid lines
        l, t, r, b = camera.visible_rect()
        start_x = int(l // spacing) * spacing
        start_y = int(t // spacing) * spacing
        for wx in range(start_x, int(r) + spacing, spacing):
            sx, _ = camera.world_to_screen(wx, 0)
            if 0 <= sx <= SW:
                sy1 = max(0, camera.world_to_screen(wx, t)[1])
                sy2 = min(SH, camera.world_to_screen(wx, b)[1])
                pygame.draw.line(self.screen, GRID_COLOR, (int(sx), int(sy1)), (int(sx), int(sy2)))
        for wy in range(start_y, int(b) + spacing, spacing):
            _, sy = camera.world_to_screen(0, wy)
            if 0 <= sy <= SH:
                sx1 = max(0, camera.world_to_screen(l, wy)[0])
                sx2 = min(SW, camera.world_to_screen(r, wy)[0])
                pygame.draw.line(self.screen, GRID_COLOR, (int(sx1), int(sy)), (int(sx2), int(sy)))

        # World border
        corners = [
            camera.world_to_screen(0, 0),
            camera.world_to_screen(config.WORLD_WIDTH, 0),
            camera.world_to_screen(config.WORLD_WIDTH, config.WORLD_HEIGHT),
            camera.world_to_screen(0, config.WORLD_HEIGHT),
        ]
        pts = [(int(x), int(y)) for x, y in corners]
        pygame.draw.lines(self.screen, (80, 100, 140), True, pts, 2)

    def _draw_oases(self, camera):
        """Draw oasis zones."""
        oases = get_oases()
        rw, rh = config.OASIS_RADIUS_W, config.OASIS_RADIUS_H
        for cx, cy in oases:
            # Screen coords of oasis rectangle
            sx1, sy1 = camera.world_to_screen(cx - rw, cy - rh)
            sx2, sy2 = camera.world_to_screen(cx + rw, cy + rh)
            sw = int(sx2 - sx1)
            sh = int(sy2 - sy1)
            if sw < 2 or sh < 2:
                continue
            # Check if visible
            if sx2 < 0 or sx1 > config.SCREEN_WIDTH or sy2 < 0 or sy1 > config.SCREEN_HEIGHT:
                continue
            zone_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            zone_surf.fill((40, 80, 60, 25))
            self.screen.blit(zone_surf, (int(sx1), int(sy1)))
            pygame.draw.rect(self.screen, (50, 100, 80, 60),
                             (int(sx1), int(sy1), sw, sh), 1)

    def _draw_food(self, items, camera):
        for item in items:
            if not camera.is_visible(item.x, item.y):
                continue
            sx, sy = camera.world_to_screen(item.x, item.y)
            ix, iy = int(sx), int(sy)
            r = max(2, int(FOOD_RADIUS * camera.zoom))
            # Glow
            gr = r * 3
            if gr > 2:
                glow_surf = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*FOOD_GLOW_COLOR, 40), (gr, gr), gr)
                self.screen.blit(glow_surf, (ix - gr, iy - gr))
            # Core
            pygame.draw.circle(self.screen, FOOD_COLOR, (ix, iy), r)
            if r > 3:
                pygame.draw.circle(self.screen, (180, 255, 200), (ix - 1, iy - 1), r // 2)

    def _draw_fishes(self, population, selected, camera):
        for fish in population:
            if camera.is_visible(fish.x, fish.y, margin=100):
                self._draw_one_fish(fish, fish is selected, camera)
        if selected and selected.alive and camera.is_visible(selected.x, selected.y, margin=200):
            self._draw_vision_rays(selected, camera)

    def _draw_one_fish(self, fish, is_selected, camera):
        alpha = 255 if fish.alive else 80
        tint = fish.color

        # Fish dimensions scaled by gene size AND camera zoom
        s = FISH_SCALE * fish.genes.size
        fs = FISH_SCALE * fish.genes.fin_size
        body_rx = 140 * s
        body_ry = 37 * s

        sx, sy = camera.world_to_screen(fish.x, fish.y)

        # Skip tiny fish (zoomed out too far)
        screen_size = body_rx * 2 * camera.zoom
        if screen_size < 3:
            # Just draw a dot
            pygame.draw.circle(self.screen, tint, (int(sx), int(sy)), max(1, int(screen_size / 2)))
            return

        # Create fish surface at world scale, then scale+rotate
        fish_w = int(body_rx * 2 + 60 * s)
        fish_h = int(body_ry * 2 + 80 * s)
        surf = pygame.Surface((fish_w * 2, fish_h * 2), pygame.SRCALPHA)
        ox, oy = fish_w, fish_h

        wave = math.sin(fish.body_wave_phase) * 3 * s

        # --- Tail stick ---
        tail_len = 100 * s
        tail_h = 7 * s
        tail_wave = math.sin(fish.body_wave_phase - 1.0) * 5 * s
        tail_rect = pygame.Rect(
            int(ox - body_rx - tail_len),
            int(oy - tail_h / 2 + tail_wave),
            int(tail_len), int(max(tail_h, 2))
        )
        pygame.draw.rect(surf, (*tint, alpha), tail_rect)

        # --- Tail fins ---
        fin_wave_top = math.sin(fish.body_wave_phase - 1.5) * 8 * s
        fin_wave_bot = math.sin(fish.body_wave_phase - 1.5 + math.pi) * 8 * s
        tail_base_x = ox - body_rx - tail_len * 0.3
        pygame.draw.polygon(surf, (*tint, alpha), [
            (tail_base_x, oy - 3 * s),
            (tail_base_x - 35 * s, oy - 30 * s + fin_wave_top),
            (tail_base_x - 5 * s, oy - 5 * s),
        ])
        pygame.draw.polygon(surf, (*tint, alpha), [
            (tail_base_x, oy + 3 * s),
            (tail_base_x - 35 * s, oy + 30 * s + fin_wave_bot),
            (tail_base_x - 5 * s, oy + 5 * s),
        ])

        # --- Body ellipse ---
        body_rect = pygame.Rect(
            int(ox - body_rx), int(oy - body_ry),
            int(body_rx * 2), int(body_ry * 2)
        )
        pygame.draw.ellipse(surf, (*tint, alpha), body_rect)
        pygame.draw.ellipse(surf, (0, 0, 0, min(alpha, 180)), body_rect, max(1, int(1 * s)))

        # --- Mating signal glow ---
        if fish.mating_signal > 0.3:
            glow_alpha = int(fish.mating_signal * 80)
            glow_r = int(body_rx * 1.5)
            pygame.draw.ellipse(surf, (255, 100, 200, glow_alpha),
                                (int(ox - glow_r), int(oy - glow_r),
                                 glow_r * 2, glow_r * 2), 2)

        # --- Side fins ---
        fin_phase_l = math.sin(fish.fin_phase) * 15 * fs
        fin_phase_r = math.sin(fish.fin_phase + math.pi * 0.7) * 15 * fs
        fin_x = ox + 10 * s
        pygame.draw.polygon(surf, (*tint, min(alpha, 200)), [
            (fin_x, oy - body_ry * 0.6),
            (fin_x - 20 * fs, oy - body_ry - 15 * fs + fin_phase_l),
            (fin_x + 15 * fs, oy - body_ry * 0.7),
        ])
        pygame.draw.polygon(surf, (*tint, min(alpha, 200)), [
            (fin_x, oy + body_ry * 0.6),
            (fin_x - 20 * fs, oy + body_ry + 15 * fs + fin_phase_r),
            (fin_x + 15 * fs, oy + body_ry * 0.7),
        ])

        # --- Mouth ---
        mouth_open_scale = 1 + fish.mouth_open * 0.8
        mouth_rx = int(3 * s * mouth_open_scale)
        mouth_ry = int(9 * s * mouth_open_scale)
        mouth_x = ox + body_rx
        mouth_rect = pygame.Rect(
            mouth_x - mouth_rx, oy - mouth_ry,
            mouth_rx * 2, mouth_ry * 2
        )
        pygame.draw.ellipse(surf, (227, 30, 36, alpha), mouth_rect)

        # --- Eyes ---
        eye_x = int(ox + body_rx * 0.55)
        eye_offset_y = int(22 * s)
        eye_rx, eye_ry = int(9 * s), int(4 * s)
        pupil_rx, pupil_ry = int(4 * s), int(2 * s)
        for ey in [oy - eye_offset_y, oy + eye_offset_y]:
            pygame.draw.ellipse(surf, (255, 255, 255, alpha),
                                (eye_x - eye_rx, ey - eye_ry, eye_rx * 2, eye_ry * 2))
            pygame.draw.ellipse(surf, (0, 0, 0, alpha),
                                (eye_x - pupil_rx, ey - pupil_ry, pupil_rx * 2, pupil_ry * 2))

        # --- Selection highlight ---
        if is_selected:
            pygame.draw.ellipse(surf, (255, 255, 100, 80),
                                (int(ox - body_rx - 5), int(oy - body_ry - 5),
                                 int(body_rx * 2 + 10), int(body_ry * 2 + 10)), 2)

        # Rotate
        angle_deg = -math.degrees(fish.angle)
        rotated = pygame.transform.rotate(surf, angle_deg)

        # Scale by camera zoom
        if abs(camera.zoom - 1.0) > 0.01:
            new_w = max(1, int(rotated.get_width() * camera.zoom))
            new_h = max(1, int(rotated.get_height() * camera.zoom))
            rotated = pygame.transform.scale(rotated, (new_w, new_h))

        rect = rotated.get_rect(center=(int(sx), int(sy)))
        self.screen.blit(rotated, rect)

    def _draw_hud(self, world, speed_mult, paused):
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        panel_w, panel_h = 280, 230
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 180))
        self.screen.blit(panel, (10, 10))

        lines = [
            f"Time: {world.time_str}",
            f"Population: {world.alive_count}  (peak: {world.alive_peak})",
            f"Births: {world.total_births}  Deaths: {world.total_deaths}",
            f"Extinctions: {world.extinctions}",
            f"Best food (alive): {world.best_alive_food}",
            f"Best food (ever): {world.best_ever_food}",
            f"Avg size: {world.avg_size:.2f}",
            f"Speed: {speed_mult}x {'[PAUSED]' if paused else ''}",
            f"Hall of Fame: {len(world.hall_of_fame.entries)}",
        ]

        y = 18
        for i, line in enumerate(lines):
            font = self.font_big if i == 0 else self.font
            text = font.render(line, True, HUD_COLOR)
            self.screen.blit(text, (20, y))
            y += 22

        hint = self.font_small.render("[F1] Help  [TAB] Config  Scroll=Zoom  MMB=Pan", True, (120, 130, 150))
        self.screen.blit(hint, (20, SH - 20))

    def _draw_message(self, message):
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        text = self.font.render(message, True, (255, 255, 100))
        rect = text.get_rect(center=(SW // 2, SH - 30))
        bg = pygame.Surface((rect.width + 20, rect.height + 10), pygame.SRCALPHA)
        bg.fill((15, 20, 35, 200))
        self.screen.blit(bg, (rect.x - 10, rect.y - 5))
        self.screen.blit(text, rect)

    def _draw_minimap(self, world, camera):
        """Draw a small minimap in the bottom-right corner."""
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        mm_w, mm_h = 180, int(180 * config.WORLD_HEIGHT / config.WORLD_WIDTH)
        mm_x = SW - mm_w - 10
        mm_y = SH - mm_h - 30

        # Background
        mm_surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
        mm_surf.fill((15, 20, 35, 160))

        # Oases
        oases = get_oases()
        rw, rh = config.OASIS_RADIUS_W, config.OASIS_RADIUS_H
        for ocx, ocy in oases:
            rx = int(ocx / config.WORLD_WIDTH * mm_w)
            ry = int(ocy / config.WORLD_HEIGHT * mm_h)
            orw = int(rw / config.WORLD_WIDTH * mm_w)
            orh = int(rh / config.WORLD_HEIGHT * mm_h)
            pygame.draw.rect(mm_surf, (40, 80, 60, 60),
                             (rx - orw, ry - orh, orw * 2, orh * 2))

        # Food dots
        for item in world.food.items:
            fx = int(item.x / config.WORLD_WIDTH * mm_w)
            fy = int(item.y / config.WORLD_HEIGHT * mm_h)
            mm_surf.set_at((max(0, min(fx, mm_w - 1)), max(0, min(fy, mm_h - 1))),
                           (100, 220, 120, 200))

        # Fish dots
        for fish in world.population:
            fx = int(fish.x / config.WORLD_WIDTH * mm_w)
            fy = int(fish.y / config.WORLD_HEIGHT * mm_h)
            fx = max(0, min(fx, mm_w - 1))
            fy = max(0, min(fy, mm_h - 1))
            c = fish.color
            pygame.draw.circle(mm_surf, (*c, 200), (fx, fy), 2)

        # Camera view rectangle
        vl, vt, vr, vb = camera.visible_rect()
        cvl = int(max(0, vl) / config.WORLD_WIDTH * mm_w)
        cvt = int(max(0, vt) / config.WORLD_HEIGHT * mm_h)
        cvr = int(min(config.WORLD_WIDTH, vr) / config.WORLD_WIDTH * mm_w)
        cvb = int(min(config.WORLD_HEIGHT, vb) / config.WORLD_HEIGHT * mm_h)
        pygame.draw.rect(mm_surf, (200, 200, 255, 120),
                         (cvl, cvt, cvr - cvl, cvb - cvt), 1)

        # Border
        pygame.draw.rect(mm_surf, (60, 80, 100), (0, 0, mm_w, mm_h), 1)

        self.screen.blit(mm_surf, (mm_x, mm_y))

    def draw_config_menu(self, menu):
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
        overlay.fill((10, 15, 30, 210))
        self.screen.blit(overlay, (0, 0))

        cx = SW // 2
        panel_w = 500
        panel_x = cx - panel_w // 2

        y = 40
        title = self.font_big.render("CONFIGURATION", True, (255, 220, 100))
        self.screen.blit(title, (cx - title.get_width() // 2, y))
        y += 30
        sub = self.font_small.render("Up/Down: select   Left/Right: adjust   Tab: close", True, (120, 135, 160))
        self.screen.blit(sub, (cx - sub.get_width() // 2, y))
        y += 30

        visible_count = 22
        start = max(0, menu.selected - visible_count // 2)
        start = min(start, max(0, len(menu.items) - visible_count))
        end = min(start + visible_count, len(menu.items))

        for i in range(start, end):
            item = menu.items[i]
            is_sel = (i == menu.selected)

            if is_sel:
                sel_bg = pygame.Surface((panel_w + 20, 22), pygame.SRCALPHA)
                sel_bg.fill((50, 70, 110, 120))
                self.screen.blit(sel_bg, (panel_x - 10, y - 2))

            label_color = (220, 230, 240) if is_sel else (150, 160, 175)
            self.screen.blit(self.font.render(item.label, True, label_color),
                             (panel_x, y))

            val_str = item.display_value()
            val_x = panel_x + panel_w - 120

            if is_sel:
                self.screen.blit(self.font.render("<", True, (100, 180, 255)),
                                 (val_x - 20, y))
                self.screen.blit(self.font.render(val_str, True, (100, 220, 255)),
                                 (val_x, y))
                vw = self.font.size(val_str)[0]
                self.screen.blit(self.font.render(">", True, (100, 180, 255)),
                                 (val_x + vw + 8, y))
            else:
                self.screen.blit(self.font.render(val_str, True, (130, 140, 160)),
                                 (val_x, y))

            if item.needs_reset:
                self.screen.blit(self.font_small.render("*", True, (255, 180, 60)),
                                 (panel_x - 15, y + 1))
            y += 24

        y = SH - 50
        if menu.pending_reset:
            warn = self.font.render("* Requires reset - will reset on close", True, (255, 180, 60))
            self.screen.blit(warn, (cx - warn.get_width() // 2, y))
        else:
            note = self.font_small.render("* = changing this value resets the simulation", True, (100, 110, 130))
            self.screen.blit(note, (cx - note.get_width() // 2, y + 5))

    def draw_help(self):
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
        overlay.fill((10, 15, 30, 220))
        self.screen.blit(overlay, (0, 0))

        cx = SW // 2
        y = 60

        title = self.font_big.render("NEUROEVOLUTION FISH - HELP", True, (255, 220, 100))
        self.screen.blit(title, (cx - title.get_width() // 2, y))
        y += 50

        sections = [
            ("SIMULATION", [
                "SPACE      Pause / Resume",
                "R          Reset (new random population)",
                "1-5        Speed: 1x, 2x, 5x, 10x, 50x",
                "ESC        Quit",
            ]),
            ("CAMERA", [
                "Scroll     Zoom in/out",
                "MMB drag   Pan the view",
                "Home       Fit entire world",
                "F          Follow selected fish",
            ]),
            ("FISH SELECTION", [
                "Click      Select a fish",
                "N          Toggle fish dashboard",
                "B          Toggle brain viewer (neural network)",
                "S          Save selected fish brain to file",
                "L          Load last saved fish into population",
            ]),
            ("SAVE & LOAD", [
                "F5         Save entire world (population + Hall of Fame)",
                "F9         Load saved world",
                "S          Save selected fish only",
                "L          Load last saved fish",
            ]),
            ("TRAINING & IMPORT", [
                "U          Open training arena (600x600)",
                "I          Import trained fish into main world",
                "           (Train first, then X to save, U to exit)",
            ]),
            ("DISPLAY & CONFIG", [
                "F1         Toggle this help screen",
                "TAB        Open config menu (adjust all parameters)",
            ]),
            ("HOW IT WORKS", [
                "Each fish has a neural network brain and physical genes.",
                "They see food/fish with 8 vision rays (345 deg) + smell.",
                "3 fins: left, right, tail. 4 memory neurons.",
                "Fish signal mating intent to nearby compatible fish.",
                "Eat enough food -> find a mate -> reproduce (crossover).",
                "Genes affect: size, color, fin size, metabolism.",
                "Food spawns in oases. If all die, Hall of Fame respawns.",
                "",
                "Hydrodynamics: fish slide easily forward but not sideways.",
                "Water pressure auto-aligns them to their velocity.",
            ]),
        ]

        for section_title, lines in sections:
            text = self.font.render(section_title, True, (100, 200, 255))
            self.screen.blit(text, (cx - 250, y))
            y += 24
            for line in lines:
                text = self.font_small.render(line, True, (180, 190, 200))
                self.screen.blit(text, (cx - 230, y))
                y += 17
            y += 10

        footer = self.font_small.render("Press F1 to close", True, (120, 130, 150))
        self.screen.blit(footer, (cx - footer.get_width() // 2, SH - 40))

    # Population history for chart
    _pop_history = []
    _pop_chart_tick = 0

    def _draw_pop_chart(self, world):
        SW = config.SCREEN_WIDTH
        if world.tick_count > self._pop_chart_tick:
            self._pop_history.append((world.alive_count, world.best_alive_food))
            self._pop_chart_tick = world.tick_count + 30
            if len(self._pop_history) > 200:
                self._pop_history = self._pop_history[-200:]

        chart_w, chart_h = 240, 100
        chart_x = SW - chart_w - 15
        chart_y = 15

        panel = pygame.Surface((chart_w + 10, chart_h + 30), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 180))
        self.screen.blit(panel, (chart_x - 5, chart_y - 5))

        title = self.font_small.render("Population / Best Food", True, HUD_COLOR)
        self.screen.blit(title, (chart_x, chart_y))

        area_y = chart_y + 18
        area_h = chart_h

        if len(self._pop_history) < 2:
            return

        pop_vals = [h[0] for h in self._pop_history]
        food_vals = [h[1] for h in self._pop_history]
        max_pop = max(max(pop_vals), 1)
        max_food = max(max(food_vals), 1)

        n = len(pop_vals)
        # Population line (blue)
        points = []
        for i, v in enumerate(pop_vals):
            px = chart_x + int(i / max(n - 1, 1) * chart_w)
            py = area_y + area_h - int(v / max_pop * area_h)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (100, 150, 220), False, points, 2)

        # Best food line (green)
        points = []
        for i, v in enumerate(food_vals):
            px = chart_x + int(i / max(n - 1, 1) * chart_w)
            py = area_y + area_h - int(v / max_food * area_h)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (100, 220, 120), False, points, 2)

        # Legend
        ly = area_y + area_h + 8
        pygame.draw.line(self.screen, (100, 150, 220), (chart_x, ly), (chart_x + 15, ly), 2)
        self.screen.blit(self.font_small.render(f"Pop ({max_pop})", True, (100, 150, 220)),
                         (chart_x + 18, ly - 6))
        pygame.draw.line(self.screen, (100, 220, 120), (chart_x + 100, ly), (chart_x + 115, ly), 2)
        self.screen.blit(self.font_small.render(f"Food ({max_food})", True, (100, 220, 120)),
                         (chart_x + 118, ly - 6))

    def _draw_vision_rays(self, fish, camera):
        sx, sy = camera.world_to_screen(fish.x, fish.y)
        fx, fy = int(sx), int(sy)
        for r in range(VISION_RAYS):
            ray_angle = fish.angle + VISION_RAY_ANGLES[r]
            ray_len = VISION_RANGE * camera.zoom

            food_d = fish.vision_food[r]
            fish_d = fish.vision_fish[r]
            fish_kin = fish.vision_fish_kin[r]
            wall_d = fish.vision_wall[r]

            closest = min(food_d, fish_d, wall_d)
            end_dist = closest * ray_len

            ex = fx + int(math.cos(ray_angle) * end_dist)
            ey = fy + int(math.sin(ray_angle) * end_dist)

            if food_d <= fish_d and food_d <= wall_d and food_d < 0.95:
                color = (80, 220, 100, 120)
            elif fish_d <= food_d and fish_d <= wall_d and fish_d < 0.95:
                # Kin = blue-green, stranger = orange
                if fish_kin > 0.5:
                    color = (60, 180, 220, 120)
                else:
                    color = (220, 160, 60, 120)
            elif wall_d < 0.95:
                color = (100, 120, 180, 100)
            else:
                color = (50, 60, 80, 40)

            ray_surf = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(ray_surf, color, (fx, fy), (ex, ey), 1)
            if closest < 0.95:
                pygame.draw.circle(ray_surf, (*color[:3], 200), (ex, ey), 3)
            self.screen.blit(ray_surf, (0, 0))

    def draw_fish_dashboard(self, fish):
        if fish is None:
            return

        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        panel_w = 300
        panel_x = SW - panel_w - 10
        panel_y = 160
        panel_h = SH - panel_y - 10

        panel = pygame.Surface((panel_w + 10, panel_h + 10), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 190))
        self.screen.blit(panel, (panel_x - 5, panel_y - 5))

        y = panel_y + 5

        self.screen.blit(self.font_big.render("Selected Fish", True, HUD_COLOR),
                         (panel_x + 5, y))
        y += 26

        # --- Stats: two columns ---
        age_secs = fish.age // 60
        col1 = [
            ("Food", str(fish.food_eaten), FOOD_COLOR),
            ("Total", str(fish.total_food), (140, 220, 140)),
            ("Energy", f"{fish.energy:.0f}", (255, 200, 80) if fish.energy > 30 else (255, 80, 60)),
            ("Age", f"{age_secs}s", (150, 160, 180)),
        ]
        col2 = [
            ("Size", f"{fish.genes.size:.2f}", (180, 150, 220)),
            ("Metab", f"{fish.genes.metabolism:.2f}", (220, 180, 100)),
            ("Kids", str(fish.children), (200, 160, 200)),
            ("Mate", f"{fish.mating_signal:.2f}", (255, 100, 200) if fish.mating_signal > 0.3 else (120, 130, 150)),
        ]
        stat_y = y
        for label, val, color in col1:
            self.screen.blit(self.font_small.render(f"{label}:", True, (120, 130, 150)),
                             (panel_x + 5, stat_y))
            self.screen.blit(self.font_small.render(val, True, color),
                             (panel_x + 55, stat_y))
            stat_y += 15
        stat_y = y
        mid_col = panel_x + panel_w // 2
        for label, val, color in col2:
            self.screen.blit(self.font_small.render(f"{label}:", True, (120, 130, 150)),
                             (mid_col + 5, stat_y))
            self.screen.blit(self.font_small.render(val, True, color),
                             (mid_col + 52, stat_y))
            stat_y += 15
        y = stat_y + 4

        # --- Motor outputs (compact) ---
        outputs = [
            ("L fin", fish.left_fin_force),
            ("R fin", fish.right_fin_force),
            ("Tail", fish.tail_force),
            ("Mouth", fish.mouth_open),
        ]
        bar_w = 120
        bar_h = 8
        bar_x = panel_x + 50
        for label, val in outputs:
            self.screen.blit(self.font_small.render(label, True, (140, 150, 170)),
                             (panel_x + 5, y - 1))
            pygame.draw.rect(self.screen, (30, 40, 60),
                             (bar_x, y, bar_w, bar_h))
            if label in ("L fin", "R fin"):
                mid = bar_x + bar_w // 2
                pygame.draw.line(self.screen, (60, 70, 90), (mid, y), (mid, y + bar_h))
                fill_w = int(val * bar_w / 2)
                if fill_w > 0:
                    pygame.draw.rect(self.screen, (80, 200, 120),
                                     (mid, y + 1, fill_w, bar_h - 2))
                elif fill_w < 0:
                    pygame.draw.rect(self.screen, (200, 100, 80),
                                     (mid + fill_w, y + 1, -fill_w, bar_h - 2))
            else:
                fill_w = int(max(0, min(1, val)) * bar_w)
                color = (80, 200, 120) if label == "Tail" else (227, 80, 60)
                if fill_w > 0:
                    pygame.draw.rect(self.screen, color,
                                     (bar_x, y + 1, fill_w, bar_h - 2))
            self.screen.blit(self.font_small.render(f"{val:+.2f}", True, (140, 150, 170)),
                             (bar_x + bar_w + 4, y - 1))
            y += 14
        y += 6

        # --- Vision radar + lateral line (left side) ---
        radar_cx = panel_x + 70
        radar_cy = y + 60
        radar_r = 52

        self.screen.blit(self.font_small.render("Vision:", True, (120, 140, 170)),
                         (panel_x + 5, y))

        pygame.draw.circle(self.screen, (25, 35, 55), (radar_cx, radar_cy), radar_r)
        pygame.draw.circle(self.screen, (50, 65, 85), (radar_cx, radar_cy), radar_r, 1)
        pygame.draw.circle(self.screen, (40, 50, 70), (radar_cx, radar_cy), radar_r // 2, 1)
        pygame.draw.circle(self.screen, (180, 180, 200), (radar_cx, radar_cy), 3)

        for r in range(VISION_RAYS):
            ray_angle = VISION_RAY_ANGLES[r]
            draw_angle = -math.pi / 2 + ray_angle

            food_d = fish.vision_food[r]
            fish_d = fish.vision_fish[r]
            fish_kin = fish.vision_fish_kin[r]
            wall_d = fish.vision_wall[r]

            if food_d < 0.95:
                dist = food_d * radar_r
                dx = radar_cx + int(math.cos(draw_angle) * dist)
                dy = radar_cy + int(math.sin(draw_angle) * dist)
                pygame.draw.circle(self.screen, (80, 220, 100), (dx, dy), 4)

            if fish_d < 0.95:
                dist = fish_d * radar_r
                dx = radar_cx + int(math.cos(draw_angle) * dist)
                dy = radar_cy + int(math.sin(draw_angle) * dist)
                fc = (60, 180, 220) if fish_kin > 0.5 else (220, 160, 60)
                pygame.draw.circle(self.screen, fc, (dx, dy), 3)

            if wall_d < 0.95:
                dist = wall_d * radar_r
                dx = radar_cx + int(math.cos(draw_angle) * dist)
                dy = radar_cy + int(math.sin(draw_angle) * dist)
                pygame.draw.circle(self.screen, (100, 120, 160), (dx, dy), 2)

            ex = radar_cx + int(math.cos(draw_angle) * radar_r)
            ey = radar_cy + int(math.sin(draw_angle) * radar_r)
            pygame.draw.line(self.screen, (40, 55, 75),
                             (radar_cx, radar_cy), (ex, ey), 1)

        # Lateral line arcs around radar
        ll_r = radar_r + 6
        sector_angles = [
            (-math.pi / 4, math.pi / 4),
            (math.pi / 4, 3 * math.pi / 4),
            (3 * math.pi / 4, 5 * math.pi / 4),
            (-3 * math.pi / 4, -math.pi / 4),
        ]
        for s in range(4):
            p = fish.lateral_pressure[s]
            m = fish.lateral_motion[s]
            if p < 0.01 and m < 0.01:
                continue
            r_col = int(120 + 135 * m)
            g_col = int(180 * p * (1 - m * 0.5))
            b_col = int(220 * p * (1 - m))
            arc_color = (min(255, r_col), min(255, g_col), min(255, b_col))
            thickness = max(2, int(p * 5))
            a_start, a_end = sector_angles[s]
            steps = 10
            for i in range(steps):
                t1 = a_start + (a_end - a_start) * i / steps
                t2 = a_start + (a_end - a_start) * (i + 1) / steps
                x1 = radar_cx + int(math.cos(-math.pi / 2 + t1) * ll_r)
                y1_arc = radar_cy + int(math.sin(-math.pi / 2 + t1) * ll_r)
                x2 = radar_cx + int(math.cos(-math.pi / 2 + t2) * ll_r)
                y2_arc = radar_cy + int(math.sin(-math.pi / 2 + t2) * ll_r)
                pygame.draw.line(self.screen, arc_color, (x1, y1_arc), (x2, y2_arc), thickness)

        # --- Senses section (right of radar) ---
        sense_x = radar_cx + radar_r + 25
        self.screen.blit(self.font_small.render("Senses:", True, (120, 140, 170)),
                         (sense_x, y))
        sense_items = [
            ("Smell", fish.smell_angle, fish.smell_dist, (100, 220, 120)),
            ("Oasis", fish.oasis_angle, fish.oasis_dist, (255, 200, 80)),
            ("Tribe", fish.tribe_dir, 1.0 - fish.tribe_count, (60, 180, 220)),
        ]
        mini_r = 18
        sense_row_y = y + 16
        for idx, (label, angle_val, dist_val, color) in enumerate(sense_items):
            cx = sense_x + 20
            cy = sense_row_y + mini_r + 2

            pygame.draw.circle(self.screen, (25, 35, 55), (cx, cy), mini_r)
            pygame.draw.circle(self.screen, (50, 65, 85), (cx, cy), mini_r, 1)

            arrow_angle = -math.pi / 2 + angle_val * math.pi
            arrow_len = mini_r * (1.0 - dist_val * 0.7)
            if dist_val < 0.99 or (label == "Tribe" and fish.tribe_count > 0.01):
                ax = cx + int(math.cos(arrow_angle) * arrow_len)
                ay = cy + int(math.sin(arrow_angle) * arrow_len)
                pygame.draw.line(self.screen, color, (cx, cy), (ax, ay), 2)
                pygame.draw.circle(self.screen, color, (ax, ay), 2)
            else:
                pygame.draw.circle(self.screen, (50, 60, 80), (cx, cy), 2)

            # Label to the right of circle
            self.screen.blit(self.font_small.render(label, True, color),
                             (cx + mini_r + 5, cy - 6))

            if label == "Tribe" and fish.tribe_count > 0.01:
                self.screen.blit(self.font_small.render(f"{fish.tribe_count:.0%}", True, color),
                                 (cx + mini_r + 5, cy + 4))

            sense_row_y += mini_r * 2 + 8

        # Legend row under radar
        leg_y = radar_cy + radar_r + 12
        leg_items = [("Food", (80, 220, 100)), ("Kin", (60, 180, 220)),
                     ("Foe", (220, 160, 60)), ("Wall", (100, 120, 160))]
        lx = panel_x + 5
        for label, color in leg_items:
            pygame.draw.circle(self.screen, color, (lx + 4, leg_y + 4), 3)
            self.screen.blit(self.font_small.render(label, True, (120, 130, 150)),
                             (lx + 10, leg_y - 2))
            lx += 55

        y = leg_y + 18

        # --- Position mini-map + compass ---
        self.screen.blit(self.font_small.render("Position:", True, (120, 140, 170)),
                         (panel_x + 5, y))
        y += 16
        map_w = int(panel_w * 0.52)
        map_h = int(map_w * config.WORLD_HEIGHT / config.WORLD_WIDTH)
        map_x = panel_x + 5

        pygame.draw.rect(self.screen, (25, 35, 55), (map_x, y, map_w, map_h))
        pygame.draw.rect(self.screen, (50, 65, 85), (map_x, y, map_w, map_h), 1)

        from food import get_oases
        oases = get_oases()
        for ox, oy in oases:
            omx = map_x + int(ox / config.WORLD_WIDTH * map_w)
            omy = y + int(oy / config.WORLD_HEIGHT * map_h)
            pygame.draw.circle(self.screen, (40, 65, 40), (omx, omy), 6, 1)

        fx_map = map_x + int(fish.x / config.WORLD_WIDTH * map_w)
        fy_map = y + int(fish.y / config.WORLD_HEIGHT * map_h)
        pygame.draw.circle(self.screen, (255, 100, 80), (fx_map, fy_map), 3)
        hdg_len = 7
        hx = fx_map + int(math.cos(fish.angle) * hdg_len)
        hy = fy_map + int(math.sin(fish.angle) * hdg_len)
        pygame.draw.line(self.screen, (255, 200, 100), (fx_map, fy_map), (hx, hy), 2)

        # Compass rose
        comp_cx = map_x + map_w + (panel_w - map_w - 10) // 2
        comp_cy = y + map_h // 2
        comp_r = min(22, (panel_w - map_w - 20) // 2)
        pygame.draw.circle(self.screen, (25, 35, 55), (comp_cx, comp_cy), comp_r)
        pygame.draw.circle(self.screen, (50, 65, 85), (comp_cx, comp_cy), comp_r, 1)
        for lbl, ang in [("N", -math.pi/2), ("E", 0), ("S", math.pi/2), ("W", math.pi)]:
            tx = comp_cx + int(math.cos(ang) * (comp_r + 7))
            ty = comp_cy + int(math.sin(ang) * (comp_r + 7))
            ls = self.font_small.render(lbl, True, (70, 80, 100))
            lr = ls.get_rect(center=(tx, ty))
            self.screen.blit(ls, lr)
        ha = fish.angle
        nhx = comp_cx + int(math.cos(ha) * (comp_r - 4))
        nhy = comp_cy + int(math.sin(ha) * (comp_r - 4))
        pygame.draw.line(self.screen, (255, 200, 100), (comp_cx, comp_cy), (nhx, nhy), 2)
        pygame.draw.circle(self.screen, (255, 200, 100), (nhx, nhy), 3)

        pos_txt = f"({fish.x:.0f}, {fish.y:.0f})"
        self.screen.blit(self.font_small.render(pos_txt, True, (100, 110, 130)),
                         (map_x, y + map_h + 2))

        y = y + map_h + 18

        # --- Memory Matrix ---
        self.screen.blit(self.font_small.render("Memory (RAM):", True, (120, 140, 170)),
                         (panel_x + 5, y))
        y += 16
        ms = len(fish.memory_matrix)
        cell_size = min(26, (panel_w - 20) // ms)
        grid_x = panel_x + (panel_w - cell_size * ms) // 2
        for r in range(ms):
            for c_idx in range(ms):
                val = fish.memory_matrix[r][c_idx]
                t = (val + 1) / 2
                red = int(clamp(t * 2 - 1, 0, 1) * 255)
                blue = int(clamp(1 - t * 2, 0, 1) * 255)
                green = int((1 - abs(val)) * 40)
                cell_c = (red, green, blue)
                cx = grid_x + c_idx * cell_size
                cy = y + r * cell_size
                pygame.draw.rect(self.screen, cell_c, (cx, cy, cell_size - 1, cell_size - 1))
                if (r, c_idx) == fish.mem_write_addr:
                    pygame.draw.rect(self.screen, (255, 255, 80), (cx, cy, cell_size - 1, cell_size - 1), 2)
                    self.screen.blit(self.font_small.render("W", True, (255, 255, 80)),
                                     (cx + 2, cy + 1))
                elif (r, c_idx) == fish.mem_read1_addr:
                    pygame.draw.rect(self.screen, (80, 255, 120), (cx, cy, cell_size - 1, cell_size - 1), 2)
                    self.screen.blit(self.font_small.render("R", True, (80, 255, 120)),
                                     (cx + 2, cy + 1))
                elif (r, c_idx) == fish.mem_read2_addr:
                    pygame.draw.rect(self.screen, (80, 220, 255), (cx, cy, cell_size - 1, cell_size - 1), 2)
                    self.screen.blit(self.font_small.render("R", True, (80, 220, 255)),
                                     (cx + 2, cy + 1))
                else:
                    pygame.draw.rect(self.screen, (50, 60, 80), (cx, cy, cell_size - 1, cell_size - 1), 1)

        leg_mem_y = y + ms * cell_size + 4
        gx = grid_x
        for label, color in [("W=Write", (255, 255, 80)), ("R=Read1", (80, 255, 120)), ("R=Read2", (80, 220, 255))]:
            self.screen.blit(self.font_small.render(label, True, color), (gx, leg_mem_y))
            gx += 75

    # --- Training arena ---
    _training_pop_history = []
    _training_chart_tick = 0

    def draw_training(self, arena, speed_mult, paused, message=None):
        """Draw the training arena view."""
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        world = arena.world

        self.screen.fill((15, 20, 35))

        # Arena dimensions on screen (centered, with padding)
        arena_size = min(SH - 80, SW - 350)
        arena_x = (SW - arena_size) // 2
        arena_y = (SH - arena_size) // 2 + 10
        scale = arena_size / 600  # 600 = training world size

        # Arena background
        arena_rect = pygame.Rect(arena_x, arena_y, arena_size, arena_size)
        pygame.draw.rect(self.screen, BG_COLOR, arena_rect)

        # Grid
        spacing = 60
        for gx in range(0, 601, spacing):
            sx = arena_x + int(gx * scale)
            pygame.draw.line(self.screen, GRID_COLOR, (sx, arena_y), (sx, arena_y + arena_size))
        for gy in range(0, 601, spacing):
            sy = arena_y + int(gy * scale)
            pygame.draw.line(self.screen, GRID_COLOR, (arena_x, sy), (arena_x + arena_size, sy))

        # Oasis zone (center)
        from food import get_oases
        oases = get_oases()
        for ocx, ocy in oases:
            rw, rh = config.OASIS_RADIUS_W, config.OASIS_RADIUS_H
            zx = arena_x + int((ocx - rw) * scale)
            zy = arena_y + int((ocy - rh) * scale)
            zw = int(rw * 2 * scale)
            zh = int(rh * 2 * scale)
            zone_surf = pygame.Surface((zw, zh), pygame.SRCALPHA)
            zone_surf.fill((40, 80, 60, 30))
            self.screen.blit(zone_surf, (zx, zy))
            pygame.draw.rect(self.screen, (50, 100, 80, 60), (zx, zy, zw, zh), 1)

        # Food
        for item in world.food.items:
            fx = arena_x + int(item.x * scale)
            fy = arena_y + int(item.y * scale)
            r = max(2, int(FOOD_RADIUS * scale))
            pygame.draw.circle(self.screen, FOOD_COLOR, (fx, fy), r)

        # Virtual camera: maps 600x600 world to arena rect on screen
        # world_to_screen: sx = (wx - cam.x) * zoom + SW/2
        # We want: sx = arena_x + wx * scale
        # So: cam.x = (SW/2 - arena_x) / scale
        from camera import Camera
        arena_cam = Camera()
        arena_cam.zoom = scale
        arena_cam.x = (config.SCREEN_WIDTH / 2 - arena_x) / scale
        arena_cam.y = (config.SCREEN_HEIGHT / 2 - arena_y) / scale

        # Clip to arena rectangle so fish don't overflow
        self.screen.set_clip(arena_rect)
        for fish in world.population:
            if fish.alive:
                self._draw_one_fish(fish, False, arena_cam)
        self.screen.set_clip(None)

        # Arena border
        pygame.draw.rect(self.screen, (100, 140, 180), arena_rect, 2)

        # --- Title ---
        title = self.font_big.render("TRAINING ARENA", True, (255, 200, 80))
        self.screen.blit(title, (SW // 2 - title.get_width() // 2, 12))
        sub = self.font_small.render("Small world - fish learn to eat here", True, (150, 160, 180))
        self.screen.blit(sub, (SW // 2 - sub.get_width() // 2, 38))

        # --- HUD (left side) ---
        hud_x = 15
        hud_y = 70
        gen_pct = int(world.gen_progress * 100)
        lines = [
            f"Generation: {world.generation}",
            f"Time: {world.time_str}",
            f"Alive: {world.alive_count}/{len(world.population)}",
            f"Best food (now): {world.best_alive_food}",
            f"Best food (gen): {world.best_gen_food}",
            f"Best food (ever): {world.best_ever_food}",
            f"Speed: {speed_mult}x {'[PAUSED]' if paused else ''}",
        ]

        panel = pygame.Surface((260, len(lines) * 22 + 50), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 180))
        self.screen.blit(panel, (hud_x - 5, hud_y - 5))

        for i, line in enumerate(lines):
            font = self.font_big if i == 0 else self.font
            self.screen.blit(font.render(line, True, HUD_COLOR), (hud_x, hud_y + i * 22))

        # Generation progress bar
        bar_y = hud_y + len(lines) * 22 + 5
        bar_w = 240
        bar_h = 14
        pygame.draw.rect(self.screen, (30, 40, 60), (hud_x, bar_y, bar_w, bar_h))
        fill_w = int(world.gen_progress * bar_w)
        pygame.draw.rect(self.screen, (80, 160, 220), (hud_x, bar_y, fill_w, bar_h))
        pct_text = self.font_small.render(f"Gen progress: {gen_pct}%", True, (180, 190, 210))
        self.screen.blit(pct_text, (hud_x + 2, bar_y + 1))

        # --- Controls hint ---
        hints = [
            "[U] Exit training     [SPACE] Pause     [1-5] Speed",
            "[X] Save top 10 genomes     [R] Reset training",
        ]
        for i, hint in enumerate(hints):
            self.screen.blit(self.font_small.render(hint, True, (120, 140, 160)),
                             (20, SH - 40 + i * 16))

        # --- Generation best chart (right side) ---
        self._draw_gen_chart(world)

        # --- Top fish list (right side) ---
        self._draw_top_fish_list(world)

        # Message
        if message:
            self._draw_message(message)

    def _draw_gen_chart(self, world):
        """Draw best food per generation chart."""
        SW = config.SCREEN_WIDTH
        history = world.gen_best_history

        chart_w, chart_h = 200, 90
        chart_x = SW - chart_w - 20
        chart_y = 70

        panel = pygame.Surface((chart_w + 10, chart_h + 30), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 180))
        self.screen.blit(panel, (chart_x - 5, chart_y - 5))

        self.screen.blit(self.font_small.render("Best Food / Generation", True, HUD_COLOR),
                         (chart_x, chart_y))

        if len(history) < 2:
            return

        area_y = chart_y + 18
        max_val = max(max(history), 1)
        n = len(history)

        points = []
        for i, v in enumerate(history[-100:]):  # last 100 gens
            px = chart_x + int(i / max(min(n, 100) - 1, 1) * chart_w)
            py = area_y + chart_h - int(v / max_val * chart_h)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (100, 220, 120), False, points, 2)

        # Label
        ly = area_y + chart_h + 6
        self.screen.blit(self.font_small.render(f"Peak: {max(history)}  Last: {history[-1]}",
                         True, (100, 220, 120)), (chart_x, ly))

    def _draw_top_fish_list(self, world):
        """Show current generation leaderboard."""
        SW = config.SCREEN_WIDTH
        lx = SW - 220
        ly = 210

        ranked = sorted(world.population, key=lambda f: f.total_food, reverse=True)

        panel = pygame.Surface((210, 20 + len(ranked) * 16), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 180))
        self.screen.blit(panel, (lx - 5, ly - 5))

        self.screen.blit(self.font.render("This Generation:", True, (200, 200, 120)), (lx, ly))
        ly += 22

        from training import TRAIN_ELITE
        for i, fish in enumerate(ranked):
            c = fish.color
            pygame.draw.rect(self.screen, c, (lx, ly + 1, 12, 12))
            label = f"#{i+1}  food:{fish.total_food}"
            if fish.alive:
                label += f"  E:{fish.energy:.0f}"
            else:
                label += "  DEAD"
            # Mark elites
            color = (255, 220, 100) if i < TRAIN_ELITE else (170, 180, 190)
            self.screen.blit(self.font_small.render(label, True, color),
                             (lx + 18, ly))
            ly += 16
