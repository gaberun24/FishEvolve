"""
Neuroevolution Fish Simulation
Press F1 for help, Tab for config menu.
"""

import sys
import os
import glob
import pygame
import config
from world import World
from renderer import Renderer
from camera import Camera
from config_menu import ConfigMenu
from training import TrainingArena
from utils import distance
from brain_visualizer import BrainVisualizer

# Absolute path for saves, handles any working directory
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Neuroevolution Fish - Aranyhal")
    clock = pygame.time.Clock()

    world = World()
    renderer = Renderer(screen)
    camera = Camera()
    cfg_menu = ConfigMenu()
    training = TrainingArena()
    brain_viz = BrainVisualizer()

    speed_mult = 1
    speed_map = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 5, pygame.K_4: 10, pygame.K_5: 50}
    paused = False
    show_nn = False
    show_brain = False
    show_help = False
    follow_selected = False

    # Pan state
    panning = False
    pan_start = (0, 0)

    # Temporary message system
    message = None
    message_timer = 0

    def set_message(msg, duration=120):
        nonlocal message, message_timer
        message = msg
        message_timer = duration

    # Key repeat for config menu
    pygame.key.set_repeat(300, 50)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ===================== TRAINING ARENA MODE =====================
            elif training.active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        # Exit training
                        training.exit()
                        renderer._training_pop_history.clear()
                        renderer._training_chart_tick = 0
                        set_message("Left training arena. Press [I] to import trained fish.", 240)

                    elif event.key == pygame.K_x:
                        # Save top 10 genomes
                        count = training.save_top_genomes(10)
                        set_message(f"Saved {count} trained genomes! Press [I] in main world to import.", 180)

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key in speed_map:
                        speed_mult = speed_map[event.key]

                    elif event.key == pygame.K_r:
                        training.reset()
                        set_message("Training reset")

                    elif event.key == pygame.K_ESCAPE:
                        training.exit()
                        renderer._training_pop_history.clear()
                        renderer._training_chart_tick = 0
                        set_message("Left training arena")

                continue  # Skip all other event handling in training mode

            # ===================== MAIN WORLD MODE =====================
            if event.type == pygame.MOUSEWHEEL:
                if not show_help and not cfg_menu.visible:
                    mx, my = pygame.mouse.get_pos()
                    factor = 1.15 if event.y > 0 else 1 / 1.15
                    camera.zoom_at(mx, my, factor)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # Middle mouse = pan
                    panning = True
                    pan_start = event.pos
                elif event.button == 1:  # Left click = select fish
                    if not show_help and not cfg_menu.visible:
                        mx, my = event.pos
                        wx, wy = camera.screen_to_world(mx, my)
                        best_fish = None
                        best_dist = 40 / camera.zoom
                        for fish in world.population:
                            d = distance(wx, wy, fish.x, fish.y)
                            if d < best_dist:
                                best_dist = d
                                best_fish = fish
                        world.selected_fish = best_fish
                        if best_fish is None:
                            follow_selected = False

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False

            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    dx = event.pos[0] - pan_start[0]
                    dy = event.pos[1] - pan_start[1]
                    camera.pan(dx, dy)
                    pan_start = event.pos

            elif event.type == pygame.KEYDOWN:

                # --- Config menu controls ---
                if cfg_menu.visible:
                    if event.key == pygame.K_TAB:
                        cfg_menu.toggle()
                        if cfg_menu.consume_reset():
                            world.reset()
                            renderer._pop_history.clear()
                            renderer._pop_chart_tick = 0
                            set_message("Config changed - simulation reset", 150)
                    elif event.key == pygame.K_UP:
                        cfg_menu.move_up()
                    elif event.key == pygame.K_DOWN:
                        cfg_menu.move_down()
                    elif event.key == pygame.K_RIGHT:
                        cfg_menu.adjust_right()
                    elif event.key == pygame.K_LEFT:
                        cfg_menu.adjust_left()
                    elif event.key == pygame.K_ESCAPE:
                        cfg_menu.toggle()
                        if cfg_menu.consume_reset():
                            world.reset()
                            renderer._pop_history.clear()
                            renderer._pop_chart_tick = 0
                            set_message("Config changed - simulation reset", 150)
                    continue

                # --- Help screen ---
                if event.key == pygame.K_F1:
                    show_help = not show_help
                elif show_help:
                    pass

                # --- Training arena ---
                elif event.key == pygame.K_u:
                    training.enter()
                    paused = False
                    set_message("Training arena! [X] save genomes, [U] exit", 180)

                # --- Import trained fish ---
                elif event.key == pygame.K_i:
                    if TrainingArena.has_trained_genomes():
                        count = TrainingArena.load_trained_fish(world)
                        set_message(f"Imported {count} trained fish!", 180)
                    else:
                        set_message("No trained genomes found. Press [U] to train first.", 120)

                # --- Normal controls ---
                elif event.key == pygame.K_TAB:
                    cfg_menu.toggle()

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    world.reset()
                    renderer._pop_history.clear()
                    renderer._pop_chart_tick = 0
                    set_message("Simulation reset")

                elif event.key == pygame.K_n:
                    show_nn = not show_nn

                elif event.key == pygame.K_b:
                    if world.selected_fish is not None:
                        show_brain = not show_brain
                        if show_brain:
                            set_message("Brain viewer - hover neurons for details", 120)
                    else:
                        set_message("Select a fish first!", 60)

                elif event.key in speed_map:
                    speed_mult = speed_map[event.key]

                elif event.key == pygame.K_ESCAPE:
                    running = False

                # --- Save/Load world ---
                elif event.key == pygame.K_F5:
                    try:
                        fn = world.save_world(SAVE_DIR)
                        pop = len([f for f in world.population if f.alive])
                        set_message(f"World saved! ({pop} fish, HoF:{len(world.hall_of_fame.entries)})", 240)
                    except Exception as e:
                        set_message(f"Save failed: {e}", 180)

                elif event.key == pygame.K_F9:
                    save_path = os.path.join(SAVE_DIR, "world_save.json")
                    if os.path.exists(save_path):
                        try:
                            world.load_world(save_path)
                            renderer._pop_history.clear()
                            renderer._pop_chart_tick = 0
                            pop = len(world.population)
                            set_message(f"World loaded! ({pop} fish, t={world.time_str})", 240)
                        except Exception as e:
                            set_message(f"Load failed: {e}", 180)
                    else:
                        set_message("No world save found! Press F5 to save first.", 120)

                elif event.key == pygame.K_HOME:
                    camera.fit_world()
                    follow_selected = False
                    set_message("Camera: fit world", 60)

                elif event.key == pygame.K_f:
                    if world.selected_fish is not None:
                        follow_selected = not follow_selected
                        set_message("Following fish" if follow_selected else "Camera free", 60)
                    else:
                        set_message("No fish selected!", 60)

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    config.POPULATION_SIZE = min(200, config.POPULATION_SIZE + 10)
                    world.reset()
                    renderer._pop_history.clear()
                    renderer._pop_chart_tick = 0
                    set_message(f"Population: {config.POPULATION_SIZE} (reset)")

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    config.POPULATION_SIZE = max(10, config.POPULATION_SIZE - 10)
                    world.reset()
                    renderer._pop_history.clear()
                    renderer._pop_chart_tick = 0
                    set_message(f"Population: {config.POPULATION_SIZE} (reset)")

                elif event.key == pygame.K_s:
                    if world.selected_fish is not None:
                        try:
                            filename = world.save_fish(world.selected_fish, SAVE_DIR)
                            set_message(f"Saved: {os.path.basename(filename)}", 180)
                        except Exception as e:
                            set_message(f"Save failed: {e}", 180)
                    else:
                        set_message("No fish selected! Click one first.")

                elif event.key == pygame.K_l:
                    try:
                        saves = sorted(glob.glob(os.path.join(SAVE_DIR, "fish_*.json")))
                        if saves:
                            loaded = world.load_fish(saves[-1])
                            worst = min(world.population, key=lambda f: f.total_food)
                            idx = world.population.index(worst)
                            world.population[idx] = loaded
                            world.selected_fish = loaded
                            set_message(f"Loaded: {os.path.basename(saves[-1])}", 180)
                        else:
                            set_message("No saved fish found!")
                    except Exception as e:
                        set_message(f"Load failed: {e}", 180)

        # --- Update ---
        if training.active:
            # Training arena tick
            if not paused:
                for _ in range(speed_mult):
                    training.world.tick()
        else:
            # Main world
            if follow_selected and world.selected_fish is not None and world.selected_fish.alive:
                camera.follow(world.selected_fish.x, world.selected_fish.y, 0.08)
            elif follow_selected and (world.selected_fish is None or not world.selected_fish.alive):
                follow_selected = False
                show_brain = False

            if not paused and not show_help and not cfg_menu.visible:
                for _ in range(speed_mult):
                    world.tick()

        # Message timer
        if message_timer > 0:
            message_timer -= 1
            if message_timer <= 0:
                message = None

        # --- Render ---
        if training.active:
            renderer.draw_training(training, speed_mult, paused, message)
        else:
            renderer.draw(world, camera, speed_mult, paused, message)
            if show_nn and world.selected_fish is not None:
                renderer.draw_fish_dashboard(world.selected_fish)
            if show_brain and world.selected_fish is not None:
                brain_viz.draw(screen, world.selected_fish)
            if show_help:
                renderer.draw_help()
            if cfg_menu.visible:
                renderer.draw_config_menu(cfg_menu)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
