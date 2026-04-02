# FishEvolve - Project Tracker

## Status: Active Development

## Completed Features

### Core Simulation
- [x] 2D world with oasis-based food zones (6000x4000, 4 oases)
- [x] Fish physics: anisotropic drag, weathervane, wall bounce, body collision
- [x] Gene system: size, metabolism, fin size, color (heritable, mutable)
- [x] Sexual reproduction with genetic compatibility (10% threshold)
- [x] Tribe-based initial population (shared gene base)
- [x] Continuous ecosystem with natural birth/death cycles
- [x] Hall of Fame: best fish remembered, respawned on extinction
- [x] Energy system: decay over time, replenished by eating

### Neural Architecture
- [x] Deep recurrent neural network `[120, 48, 24, 12]`
- [x] Recurrent feedback: Hidden1 (48 neurons) fed back as input
- [x] 7,284 evolvable weights
- [x] Addressed memory matrix (4x4 RAM with write + 2 read heads)
- [x] Activation storage for visualization

### Sensory System
- [x] 12-ray vision with 4 channels (food, fish dist, kin recognition, wall)
- [x] Front + back blind spots (realistic fish vision)
- [x] Lateral line organ: 4-sector pressure + motion sensing (360 degrees)
- [x] Smell: noisy direction + distance to nearest food
- [x] Self-awareness: speed, angular velocity, energy
- [x] Mating signal detection
- [x] Spatial awareness: world position, compass heading
- [x] Oasis homing: direction + distance to nearest food zone
- [x] Tribe sense: nearby kin count + direction

### Visualization
- [x] Smooth camera with zoom, pan, follow mode
- [x] Fish dashboard (N key): stats, motor bars, vision radar, lateral line arcs, senses compasses, position mini-map, compass rose, memory matrix heatmap
- [x] Brain viewer (B key): full NN visualization with grouped inputs, activation colors, weight connections on hover, recurrent arc, tooltips
- [x] Population/food chart (real-time sparkline)
- [x] World minimap in dashboard

### UI & Controls
- [x] Speed control (1x-50x)
- [x] Pause/resume
- [x] Config menu (Tab): live parameter tweaking
- [x] Help screen (F1)
- [x] Fish selection by click
- [x] Save/load individual fish (S/L)
- [x] Save/load entire world (F5/F9)
- [x] Training arena (U): isolated generational evolution
- [x] Import trained fish (I): inject trained brains into main world
- [x] Population size adjustment (+/-)

## Possible Future Enhancements

### Simulation
- [ ] Predator fish species (carnivores that hunt herbivores)
- [ ] Environmental hazards (currents, temperature zones)
- [ ] Day/night cycle affecting vision range
- [ ] Multiple food types with different nutritional values
- [ ] Egg laying instead of instant offspring
- [ ] Territorial behavior zones

### Neural Architecture
- [ ] Attention mechanism for selective focus
- [ ] Neuromodulation (outputs that modify learning/behavior)
- [ ] Communication between fish (sound signals)
- [ ] Larger memory matrix options (8x8 for complex behaviors)

### Visualization
- [ ] Phylogenetic tree (family tree of the population)
- [ ] Gene pool diversity heatmap
- [ ] Real-time fitness graphs per lineage
- [ ] Replay system (record and playback interesting moments)
- [ ] Screenshot/video export

### Technical
- [ ] GPU acceleration for neural network forward pass
- [ ] Spatial partitioning for collision (quadtree)
- [ ] Multi-threaded fish updates
- [ ] Web version (WASM/JS port)
- [ ] Headless mode for overnight training

## Tech Stack
- **Python 3.10+**
- **Pygame 2.x** — rendering, input, window management
- **NumPy** — neural network matrix operations, genetics

## File Structure
```
fisher/
  main.py              Entry point, game loop
  config.py            All constants and computed values
  fish.py              Fish model (senses, physics, memory, reproduction)
  neural_network.py    Deep recurrent NN class
  genes.py             Gene system with compatibility
  food.py              Oasis food manager
  world.py             Ecosystem manager, save/load
  camera.py            Zoom/pan camera
  renderer.py          All rendering (fish, food, HUD, dashboard)
  brain_visualizer.py  Neural network visualization
  config_menu.py       Live config tweaking UI
  training.py          Training arena (generational evolution)
  utils.py             Math helpers
  genetics.py          Selection/crossover helpers
  assets/
    aranyhal.svg       Reference goldfish SVG
```
