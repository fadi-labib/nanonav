#!/usr/bin/env python3
"""
Visual A* Test

Run a real A* pathfinding and visualize the result.

Usage:
    python scripts/test_astar_visual.py
    python scripts/test_astar_visual.py --seed 42 --size 8 --density 0.25
    python scripts/test_astar_visual.py --save results/plots/astar_test.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from trm_nav.map_generator import generate_solvable_map
from trm_nav.a_star import astar_path, path_to_actions, ACTION_NAMES
from trm_nav.visualize import plot_grid, plot_action_sequence, print_grid_ascii


def main():
    parser = argparse.ArgumentParser(description="Visual A* Test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--size", type=int, default=8, help="Grid size")
    parser.add_argument("--density", type=float, default=0.2, help="Obstacle density")
    parser.add_argument("--save", type=str, default=None, help="Save path for image")
    parser.add_argument("--no-plot", action="store_true", help="ASCII only, no matplotlib")
    args = parser.parse_args()

    print("=" * 50)
    print("A* Pathfinding Test")
    print("=" * 50)
    print(f"Grid size: {args.size}x{args.size}")
    print(f"Obstacle density: {args.density}")
    print(f"Seed: {args.seed}")
    print()

    # Generate map
    print("Generating map...")
    grid, start, goal = generate_solvable_map(
        size=args.size,
        obstacle_density=args.density,
        seed=args.seed
    )

    # Run A*
    print("Running A*...")
    path = astar_path(grid, start, goal)
    actions = path_to_actions(path)

    # Results
    print()
    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Start:       {start}")
    print(f"Goal:        {goal}")
    print(f"Path length: {len(path)} positions ({len(actions)} moves)")
    print(f"Obstacles:   {grid.sum()} / {args.size * args.size} cells")
    print()
    print("Actions:")
    print("  " + " → ".join(ACTION_NAMES[a] for a in actions))
    print()

    # ASCII visualization
    print("ASCII Grid:")
    print_grid_ascii(grid, start, goal, path)

    # Matplotlib visualization
    if not args.no_plot:
        print()
        print("Displaying plot... (close window to exit)")

        title = f"A* Pathfinding | {len(actions)} moves | Seed: {args.seed}"
        plot_grid(
            grid, start, goal,
            path=path,
            title=title,
            save_path=args.save,
            show=True
        )

        if args.save:
            print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
