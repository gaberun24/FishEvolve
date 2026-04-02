"""Brain Visualizer - beautiful neural network visualization overlay.

Press B to toggle. Shows all neurons, weights, activations and recurrent
connections for the selected fish.
"""
import math
import pygame
import numpy as np
import config
from utils import clamp

# Input group definitions: (name, start_index, count, color)
INPUT_GROUPS = [
    ("Vision",        0, 48, (80, 220, 100)),
    ("Lat. Line",    48,  8, (100, 180, 220)),
    ("Smell",        56,  2, (220, 200, 80)),
    ("Self",         58,  3, (180, 150, 220)),
    ("Mating",       61,  1, (255, 100, 200)),
    ("Position",     62,  2, (150, 200, 180)),
    ("Compass",      64,  2, (200, 200, 130)),
    ("Oasis",        66,  2, (255, 200, 80)),
    ("Tribe",        68,  2, (60, 180, 220)),
    ("Mem Read",     70,  2, (80, 255, 180)),
    ("Recurrent",    72, config.RECURRENT_SIZE, (160, 100, 255)),
]

OUTPUT_GROUPS = [
    ("Motor",         0, 5, (80, 200, 120)),
    ("Mem Write",     5, 3, (255, 255, 80)),
    ("Mem Read",      8, 4, (80, 220, 255)),
]

OUTPUT_LABELS = [
    "L Fin", "R Fin", "Tail", "Mouth", "Mate",
    "W Row", "W Col", "W Val",
    "R1 Row", "R1 Col", "R2 Row", "R2 Col",
]

LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Output"]


def _activation_color(val):
    """Map activation [-1, 1] to color: blue -> dark -> orange."""
    t = (val + 1.0) / 2.0  # 0..1
    if t < 0.5:
        # Blue to dark
        f = t / 0.5
        return (int(30 * f), int(30 * f), int(180 * (1 - f) + 40 * f))
    else:
        # Dark to orange
        f = (t - 0.5) / 0.5
        return (int(40 + 215 * f), int(30 + 140 * f), int(40 * (1 - f)))


def _weight_color(val, alpha=255):
    """Map weight to color: negative=red, positive=green, zero=gray."""
    v = clamp(val, -2, 2) / 2.0  # normalize to [-1, 1]
    if v > 0:
        return (40, int(80 + 175 * v), 60, alpha)
    else:
        return (int(80 + 175 * abs(v)), 40, 40, alpha)


class BrainVisualizer:
    """Full-screen overlay showing neural network internals."""

    def __init__(self):
        self._fonts_ready = False
        self.font_title = None
        self.font_label = None
        self.font_tiny = None
        self.hovered_neuron = None  # (layer, index) or None
        self._weight_surface = None
        self._last_fish_id = None

    def _init_fonts(self):
        if not self._fonts_ready:
            self.font_title = pygame.font.SysFont('Consolas', 22, bold=True)
            self.font_label = pygame.font.SysFont('Consolas', 13)
            self.font_tiny = pygame.font.SysFont('Consolas', 10)
            self._fonts_ready = True

    def draw(self, screen, fish):
        """Draw full brain visualization overlay."""
        self._init_fonts()
        SW, SH = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        brain = fish.brain

        if not brain.activations:
            return

        # Dark overlay background
        overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
        overlay.fill((10, 15, 25, 235))
        screen.blit(overlay, (0, 0))

        # Layout constants
        margin_left = 160
        margin_right = 100
        margin_top = 65
        margin_bottom = 40
        usable_w = SW - margin_left - margin_right
        usable_h = SH - margin_top - margin_bottom
        n_layers = len(brain.topology)

        # Calculate node positions for each layer
        node_pos = []  # list of list of (x, y)
        col_xs = []

        for li in range(n_layers):
            size = brain.topology[li]
            x = margin_left + int(li / (n_layers - 1) * usable_w) if n_layers > 1 else SW // 2
            col_xs.append(x)

            if li == 0:
                # Input layer: grouped layout
                positions = self._layout_input_layer(x, margin_top, usable_h)
            elif li == n_layers - 1:
                # Output layer: grouped layout
                positions = self._layout_output_layer(x, margin_top, usable_h)
            else:
                # Hidden layers: even spacing
                positions = self._layout_even(x, margin_top, usable_h, size)
            node_pos.append(positions)

        # Check mouse hover
        mx, my = pygame.mouse.get_pos()
        self.hovered_neuron = None
        for li, positions in enumerate(node_pos):
            for ni, (nx, ny) in enumerate(positions):
                if abs(mx - nx) < 8 and abs(my - ny) < 8:
                    self.hovered_neuron = (li, ni)
                    break

        # Draw connections
        self._draw_connections(screen, brain, node_pos)

        # Draw recurrent feedback arc
        self._draw_recurrent_arc(screen, node_pos, brain)

        # Draw nodes
        self._draw_nodes(screen, brain, node_pos)

        # Draw group labels (left of input, right of output)
        self._draw_input_labels(screen, node_pos[0], margin_left)
        self._draw_output_labels(screen, node_pos[-1])

        # Layer headers
        for li in range(n_layers):
            name = LAYER_NAMES[li] if li < len(LAYER_NAMES) else f"Layer {li}"
            size = brain.topology[li]
            txt = f"{name} ({size})"
            surf = self.font_label.render(txt, True, (140, 150, 180))
            rect = surf.get_rect(centerx=col_xs[li], top=margin_top - 22)
            screen.blit(surf, rect)

        # Title bar
        title = self.font_title.render("BRAIN VIEWER", True, (255, 220, 100))
        screen.blit(title, (20, 15))

        weights_total = len(brain.get_flat_weights())
        info = self.font_label.render(
            f"Topology: {brain.topology}  |  Recurrent: {brain.recurrent_size}  |  "
            f"Weights: {weights_total:,}  |  [B] close  |  Hover neuron for connections",
            True, (100, 110, 140))
        screen.blit(info, (20, 42))

        # Hover info
        if self.hovered_neuron:
            self._draw_hover_info(screen, brain, node_pos)

    # --- Layout methods ---

    def _layout_input_layer(self, x, top, height):
        """Layout input neurons grouped by sense type."""
        positions = []
        total = sum(g[2] for g in INPUT_GROUPS)
        # Calculate total visual height needed with gaps
        gap = 6  # gap between groups
        node_spacing = max(3, min(8, (height - gap * len(INPUT_GROUPS)) / total))

        y = top + 10
        for name, start, count, color in INPUT_GROUPS:
            for i in range(count):
                positions.append((x, y))
                y += node_spacing
            y += gap  # gap between groups

        # Center vertically
        actual_h = y - top - 10
        offset = (height - actual_h) / 2
        if offset > 0:
            positions = [(px, py + offset) for px, py in positions]

        return positions

    def _layout_output_layer(self, x, top, height):
        """Layout output neurons grouped by function."""
        positions = []
        total = sum(g[2] for g in OUTPUT_GROUPS)
        spacing = min(30, height / (total + len(OUTPUT_GROUPS)))
        y = top + (height - total * spacing) / 2

        for name, start, count, color in OUTPUT_GROUPS:
            for i in range(count):
                positions.append((x, y))
                y += spacing
            y += 10  # gap between groups

        return positions

    def _layout_even(self, x, top, height, count):
        """Evenly space neurons in a column."""
        positions = []
        if count <= 1:
            return [(x, top + height // 2)]
        spacing = min(18, height / (count + 1))
        total_h = spacing * (count - 1)
        start_y = top + (height - total_h) / 2
        for i in range(count):
            positions.append((x, start_y + i * spacing))
        return positions

    # --- Drawing methods ---

    def _draw_connections(self, screen, brain, node_pos):
        """Draw weight connections between layers."""
        conn_surf = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)

        for li in range(len(brain.weights)):
            w = brain.weights[li]  # shape: (n_in, n_out)
            src = node_pos[li]
            dst = node_pos[li + 1]

            if self.hovered_neuron:
                hl, hi = self.hovered_neuron
                # Only draw connections touching the hovered neuron
                if hl == li:
                    # Hovered is in source layer - draw all outgoing
                    for j in range(w.shape[1]):
                        val = w[hi, j]
                        if abs(val) < 0.02:
                            continue
                        thick = max(1, int(abs(val) * 2.5))
                        color = _weight_color(val, min(255, int(abs(val) * 180 + 60)))
                        pygame.draw.line(conn_surf, color,
                                         (int(src[hi][0]), int(src[hi][1])),
                                         (int(dst[j][0]), int(dst[j][1])), thick)
                elif hl == li + 1:
                    # Hovered is in dest layer - draw all incoming
                    for i in range(w.shape[0]):
                        val = w[i, hi]
                        if abs(val) < 0.02:
                            continue
                        thick = max(1, int(abs(val) * 2.5))
                        color = _weight_color(val, min(255, int(abs(val) * 180 + 60)))
                        pygame.draw.line(conn_surf, color,
                                         (int(src[i][0]), int(src[i][1])),
                                         (int(dst[hi][0]), int(dst[hi][1])), thick)
            else:
                # No hover: draw strong connections only
                threshold = 0.4
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        val = w[i, j]
                        if abs(val) < threshold:
                            continue
                        alpha = min(60, int(abs(val) * 30))
                        color = _weight_color(val, alpha)
                        pygame.draw.line(conn_surf, color,
                                         (int(src[i][0]), int(src[i][1])),
                                         (int(dst[j][0]), int(dst[j][1])), 1)

        screen.blit(conn_surf, (0, 0))

    def _draw_recurrent_arc(self, screen, node_pos, brain):
        """Draw the recurrent feedback path from hidden1 back to input."""
        if brain.recurrent_size == 0:
            return

        # Source: hidden1 column center
        h1_positions = node_pos[1]
        if not h1_positions:
            return
        h1_top = h1_positions[0][1]
        h1_bot = h1_positions[-1][1]
        h1_x = h1_positions[0][0]
        h1_cy = (h1_top + h1_bot) / 2

        # Destination: recurrent section of input
        inp_positions = node_pos[0]
        rec_start = config.SENSORY_INPUTS
        if rec_start >= len(inp_positions):
            return
        rec_top = inp_positions[rec_start][1]
        rec_bot = inp_positions[-1][1]
        rec_x = inp_positions[rec_start][0]
        rec_cy = (rec_top + rec_bot) / 2

        # Draw a curved arrow going below
        arc_y = min(config.SCREEN_HEIGHT - 30, max(h1_bot, rec_bot) + 40)
        color = (160, 100, 255, 120)

        arc_surf = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)

        # Draw dashed curved path: h1_center -> down -> left -> up -> rec_center
        points = []
        steps = 40
        for t in range(steps + 1):
            f = t / steps
            if f < 0.25:
                # Go down from h1
                ff = f / 0.25
                px = h1_x + 15
                py = h1_cy + (arc_y - h1_cy) * ff
            elif f < 0.75:
                # Go left along bottom
                ff = (f - 0.25) / 0.5
                px = h1_x + 15 - (h1_x + 15 - rec_x + 15) * ff
                py = arc_y
            else:
                # Go up to rec
                ff = (f - 0.75) / 0.25
                px = rec_x - 15
                py = arc_y - (arc_y - rec_cy) * ff
            points.append((int(px), int(py)))

        # Draw dashed
        for i in range(0, len(points) - 1, 2):
            pygame.draw.line(arc_surf, color, points[i], points[min(i + 1, len(points) - 1)], 2)

        # Arrowhead at end
        end = points[-1]
        pygame.draw.polygon(arc_surf, (160, 100, 255, 180), [
            (end[0] + 8, end[1]),
            (end[0] - 2, end[1] - 5),
            (end[0] - 2, end[1] + 5),
        ])

        # Label
        label = self.font_tiny.render("recurrent", True, (160, 100, 255))
        arc_surf.blit(label, ((h1_x + rec_x) // 2 - 20, arc_y + 4))

        screen.blit(arc_surf, (0, 0))

    def _draw_nodes(self, screen, brain, node_pos):
        """Draw neuron circles colored by activation."""
        for li, positions in enumerate(node_pos):
            activations = brain.activations[li] if li < len(brain.activations) else None

            for ni, (nx, ny) in enumerate(positions):
                # Determine color from activation
                if activations is not None and ni < len(activations):
                    act = float(activations[ni])
                    color = _activation_color(act)
                    radius = max(3, min(7, 3 + int(abs(act) * 4)))
                else:
                    color = (40, 45, 60)
                    radius = 3

                # Input layer: use group color for border
                if li == 0:
                    group_color = self._get_input_group_color(ni)
                    pygame.draw.circle(screen, color, (int(nx), int(ny)), radius)
                    pygame.draw.circle(screen, group_color, (int(nx), int(ny)), radius, 1)
                elif li == len(node_pos) - 1:
                    group_color = self._get_output_group_color(ni)
                    pygame.draw.circle(screen, color, (int(nx), int(ny)), radius)
                    pygame.draw.circle(screen, group_color, (int(nx), int(ny)), radius, 1)
                else:
                    pygame.draw.circle(screen, color, (int(nx), int(ny)), radius)
                    pygame.draw.circle(screen, (60, 70, 90), (int(nx), int(ny)), radius, 1)

                # Highlight hovered
                if self.hovered_neuron == (li, ni):
                    pygame.draw.circle(screen, (255, 255, 255), (int(nx), int(ny)), radius + 3, 2)

    def _draw_input_labels(self, screen, positions, margin_left):
        """Draw input group labels to the left of input column."""
        for name, start, count, color in INPUT_GROUPS:
            if start >= len(positions) or start + count - 1 >= len(positions):
                continue
            top_y = positions[start][1]
            bot_y = positions[start + count - 1][1]
            mid_y = (top_y + bot_y) / 2
            x = positions[start][0]

            # Bracket line
            bx = x - 15
            pygame.draw.line(screen, (*color, 100), (bx, int(top_y)), (bx, int(bot_y)), 1)
            pygame.draw.line(screen, (*color, 100), (bx, int(top_y)), (bx + 4, int(top_y)), 1)
            pygame.draw.line(screen, (*color, 100), (bx, int(bot_y)), (bx + 4, int(bot_y)), 1)

            # Label
            label = self.font_tiny.render(f"{name} ({count})", True, color)
            rect = label.get_rect(right=bx - 4, centery=int(mid_y))
            screen.blit(label, rect)

    def _draw_output_labels(self, screen, positions):
        """Draw output labels to the right of output column."""
        for i, (nx, ny) in enumerate(positions):
            if i < len(OUTPUT_LABELS):
                label = OUTPUT_LABELS[i]
                group_color = self._get_output_group_color(i)
                surf = self.font_tiny.render(label, True, group_color)
                screen.blit(surf, (int(nx) + 12, int(ny) - 5))

        # Group brackets
        for name, start, count, color in OUTPUT_GROUPS:
            if start >= len(positions) or start + count - 1 >= len(positions):
                continue
            top_y = positions[start][1]
            bot_y = positions[start + count - 1][1]
            x = positions[start][0]
            bx = x + 70
            pygame.draw.line(screen, (*color, 100), (bx, int(top_y)), (bx, int(bot_y)), 1)

    def _draw_hover_info(self, screen, brain, node_pos):
        """Draw tooltip for hovered neuron."""
        li, ni = self.hovered_neuron
        layer_name = LAYER_NAMES[li] if li < len(LAYER_NAMES) else f"Layer {li}"
        act = brain.activations[li][ni] if li < len(brain.activations) and ni < len(brain.activations[li]) else 0

        # Build info text
        lines = [f"{layer_name} neuron #{ni}", f"Activation: {act:+.4f}"]

        if li == 0:
            # Input neuron - identify group
            for name, start, count, color in INPUT_GROUPS:
                if start <= ni < start + count:
                    sub_idx = ni - start
                    lines.append(f"Group: {name}[{sub_idx}]")
                    break
        elif li == len(brain.topology) - 1:
            # Output neuron
            if ni < len(OUTPUT_LABELS):
                lines.append(f"Output: {OUTPUT_LABELS[ni]}")

        # Count connections
        if li > 0:
            w_in = brain.weights[li - 1]
            n_strong_in = np.sum(np.abs(w_in[:, ni]) > 0.1)
            lines.append(f"Strong inputs: {n_strong_in}")
        if li < len(brain.weights):
            w_out = brain.weights[li]
            n_strong_out = np.sum(np.abs(w_out[ni, :]) > 0.1)
            lines.append(f"Strong outputs: {n_strong_out}")

        # Draw tooltip box
        mx, my = pygame.mouse.get_pos()
        tw = 200
        th = len(lines) * 16 + 10
        tx = min(mx + 15, config.SCREEN_WIDTH - tw - 10)
        ty = min(my + 15, config.SCREEN_HEIGHT - th - 10)

        tip_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        tip_surf.fill((20, 25, 40, 220))
        pygame.draw.rect(tip_surf, (80, 90, 120), (0, 0, tw, th), 1)

        for i, line in enumerate(lines):
            color = (200, 210, 230) if i == 0 else (150, 160, 180)
            surf = self.font_tiny.render(line, True, color)
            tip_surf.blit(surf, (8, 5 + i * 16))

        screen.blit(tip_surf, (tx, ty))

    # --- Helpers ---

    def _get_input_group_color(self, neuron_idx):
        for name, start, count, color in INPUT_GROUPS:
            if start <= neuron_idx < start + count:
                return color
        return (60, 70, 90)

    def _get_output_group_color(self, neuron_idx):
        for name, start, count, color in OUTPUT_GROUPS:
            if start <= neuron_idx < start + count:
                return color
        return (60, 70, 90)
