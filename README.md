# FishEvolve - Neuroevolution Fish Simulation

A real-time 2D simulation where fish controlled by **deep recurrent neural networks** evolve through **neuroevolution** to survive, eat, and reproduce. Watch as generations of fish develop increasingly sophisticated behaviors — from blind wandering to coordinated food-seeking strategies.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.x-green) ![NumPy](https://img.shields.io/badge/NumPy-orange)

## Features

### Neural Architecture
- **Deep Recurrent Network** `[120, 48, 24, 12]` — two hidden layers with recurrent feedback from Hidden1, giving fish implicit working memory ("thoughts" between ticks)
- **7,284 evolvable weights** — enough for complex behavior, light enough for real-time evolution
- **Addressed Memory Matrix (RAM)** — 4x4 external memory with write head and 2 read heads, evolved alongside the brain

### Rich Sensory System
| Sense | Inputs | Description |
|-------|--------|-------------|
| **Vision** | 48 | 12 rays x 4 channels (food, fish distance, kin recognition, wall) |
| **Lateral Line** | 8 | 4 sectors x 2 channels (pressure + motion), 360 degrees including blind spots |
| **Smell** | 2 | Rough direction + distance to nearest food (noisy, long range) |
| **Self** | 3 | Speed, angular velocity, energy fraction |
| **Mating Signal** | 1 | Nearby compatible fish broadcasting willingness to mate |
| **Position** | 2 | Normalized world coordinates |
| **Compass** | 2 | sin/cos of heading direction |
| **Oasis Homing** | 2 | Angle + distance to nearest food oasis |
| **Tribe Sense** | 2 | Nearby kin count + average direction to kin |
| **Memory Read** | 2 | Values from 2 read heads addressing the memory matrix |
| **Recurrent State** | 48 | Hidden layer 1 fed back from previous tick |

### Ecosystem
- **Sexual reproduction** with genetic compatibility — fish must find a mate with similar genes (size, color)
- **Gene system** — size, metabolism, fin size, color are heritable and affect physics
- **Oasis-based food** — food spawns in 4 oasis zones, fish must navigate to find them
- **Hall of Fame** — best fish ever are remembered and respawned on population collapse
- **Continuous ecosystem** — no fixed generations, fish live, reproduce, and die naturally

### Visualization
- **Real-time dashboard** (N key) — vision radar with kin recognition, lateral line arcs, smell/oasis/tribe compasses, world mini-map, compass rose, memory matrix heatmap with read/write head markers
- **Brain viewer** (B key) — full neural network visualization with 120 input neurons (color-coded by sense group), all hidden layers, output labels, weight connections on hover, recurrent feedback arc, and activation tooltips
- **Training arena** (U key) — isolated generational evolution environment for rapid learning

## Screenshots

*Press N on a selected fish to see the dashboard. Press B to see the brain.*

## Installation

```bash
# Requirements
pip install pygame numpy

# Run
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| **Click** | Select a fish |
| **F** | Follow selected fish |
| **N** | Toggle fish dashboard |
| **B** | Toggle brain viewer |
| **1-5** | Speed (1x, 2x, 5x, 10x, 50x) |
| **Space** | Pause/Resume |
| **Tab** | Config menu (tweak all parameters live) |
| **Home** | Fit camera to world |
| **Scroll** | Zoom |
| **MMB drag** | Pan |
| **R** | Reset simulation |
| **S** | Save selected fish |
| **L** | Load last saved fish |
| **F5** | Save entire world |
| **F9** | Load saved world |
| **U** | Enter training arena |
| **I** | Import trained fish to main world |
| **+/-** | Adjust population size |
| **F1** | Help screen |
| **Esc** | Quit |

## Architecture

```
main.py              Game loop, input handling, state management
config.py            All tunable constants (world, physics, NN, genetics)
fish.py              Fish model: senses, physics, reproduction, memory
neural_network.py    Deep recurrent neural network with activation storage
genes.py             Heritable traits: size, metabolism, fin size, color
food.py              Oasis-based food spawning and management
world.py             Ecosystem: population, hall of fame, save/load
camera.py            Zoom/pan camera with follow mode
renderer.py          Full rendering: fish, food, HUD, dashboard, radar
brain_visualizer.py  Neural network visualization overlay
config_menu.py       Live parameter tweaking UI
training.py          Isolated training arena with generational evolution
utils.py             Math helpers (angle, distance, clamp)
```

## How It Works

1. **Each fish** has a brain (deep recurrent NN), genes, and a 4x4 memory matrix
2. **Every tick**, the fish perceives the world through 72 sensory inputs + 48 recurrent state = 120 total inputs
3. **The brain outputs** 12 values: 5 motor (fins, tail, mouth, mating signal), 3 memory write (row, col, value), 4 memory read (2 heads x row, col)
4. **Fish that eat** enough food can find compatible mates and reproduce — offspring inherit a crossover of both parents' brains and genes with mutations
5. **Natural selection** — fish that can't find food die. Fish that eat well and find mates pass on their neural architecture.
6. Over time, the population evolves increasingly effective foraging, navigation, and social behaviors.

## License

MIT
