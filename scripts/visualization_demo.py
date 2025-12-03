#!/usr/bin/env python3
"""
Visualization Demo

Showcases all visualization capabilities of NanoNav.

Usage:
    python scripts/visualization_demo.py
    python scripts/visualization_demo.py --save  # Save images to results/plots/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from trm_nav.map_generator import generate_solvable_map
from trm_nav.a_star import astar_path, path_to_actions, ACTION_NAMES
from trm_nav.visualize import (
    plot_grid,
    plot_action_sequence,
    plot_comparison,
    plot_training_history,
    plot_benchmark_results,
    print_grid_ascii,
    create_navigation_animation,
)


def demo_ascii_grid():
    """Demo 1: ASCII grid in terminal."""
    print("\n" + "=" * 60)
    print("DEMO 1: ASCII Grid Visualization")
    print("=" * 60)

    grid, start, goal = generate_solvable_map(size=8, seed=42)
    optimal_path = astar_path(grid, start, goal)
    actions = path_to_actions(optimal_path)

    print("\nLegend: 🟢=Start, 🎯=Goal, 🟡=Path, ⬛=Obstacle, ⬜=Free")
    print_grid_ascii(grid, start, goal, optimal_path)

    print(f"\nStart: {start}")
    print(f"Goal:  {goal}")
    print(f"Path:  {len(optimal_path)} positions, {len(actions)} moves")
    print(f"Actions: {' → '.join(ACTION_NAMES[a] for a in actions)}")


def demo_grid_plot(save: bool = False):
    """Demo 2: Matplotlib grid with path."""
    print("\n" + "=" * 60)
    print("DEMO 2: Grid Plot with Path")
    print("=" * 60)

    grid, start, goal = generate_solvable_map(size=8, seed=123)
    optimal_path = astar_path(grid, start, goal)

    save_path = "results/plots/demo_grid.png" if save else None

    print("\nDisplaying grid with A* optimal path...")
    print("(Close the window to continue)")

    plot_grid(
        grid, start, goal,
        path=optimal_path,
        title="Grid World with A* Path",
        save_path=save_path
    )


def demo_action_sequence(save: bool = False):
    """Demo 3: Action sequence visualization."""
    print("\n" + "=" * 60)
    print("DEMO 3: Action Sequence")
    print("=" * 60)

    grid, start, goal = generate_solvable_map(size=8, seed=456)
    optimal_path = astar_path(grid, start, goal)
    actions = path_to_actions(optimal_path)

    save_path = "results/plots/demo_actions.png" if save else None

    print(f"\nActions: {[ACTION_NAMES[a] for a in actions]}")
    print("Displaying action sequence as colored boxes...")

    plot_action_sequence(
        actions,
        title=f"Action Sequence ({len(actions)} steps)",
        save_path=save_path
    )


def demo_path_comparison(save: bool = False):
    """Demo 4: Side-by-side path comparison."""
    print("\n" + "=" * 60)
    print("DEMO 4: Path Comparison (A* vs Simulated TRM)")
    print("=" * 60)

    grid, start, goal = generate_solvable_map(size=8, seed=789)
    optimal_path = astar_path(grid, start, goal)

    # Simulate a suboptimal TRM path (adds some detours)
    trm_path = list(optimal_path)
    # Add a small detour if possible
    if len(trm_path) > 3:
        mid = len(trm_path) // 2
        mid_pos = trm_path[mid]
        # Try to add a detour
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (mid_pos[0] + dr, mid_pos[1] + dc)
            if (0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8 and
                grid[new_pos[0], new_pos[1]] == 0 and new_pos not in trm_path):
                trm_path.insert(mid + 1, new_pos)
                trm_path.insert(mid + 2, mid_pos)
                break

    save_path = "results/plots/demo_comparison.png" if save else None

    print(f"\nA* path length: {len(optimal_path)}")
    print(f"TRM path length: {len(trm_path)} (simulated with detour)")
    print("Displaying side-by-side comparison...")

    plot_comparison(
        grid, start, goal,
        trm_path, optimal_path,
        title="Path Comparison",
        save_path=save_path
    )


def demo_training_history(save: bool = False):
    """Demo 5: Training history plots."""
    print("\n" + "=" * 60)
    print("DEMO 5: Training History")
    print("=" * 60)

    # Simulate training history
    epochs = 50
    history = {
        'train_loss': [2.0 * np.exp(-0.05 * i) + 0.3 + 0.1 * np.random.randn() for i in range(epochs)],
        'val_loss': [2.2 * np.exp(-0.04 * i) + 0.35 + 0.15 * np.random.randn() for i in range(epochs)],
        'train_acc': [1 - 0.8 * np.exp(-0.06 * i) + 0.02 * np.random.randn() for i in range(epochs)],
        'val_acc': [1 - 0.85 * np.exp(-0.05 * i) + 0.03 * np.random.randn() for i in range(epochs)],
    }

    # Clip values
    history['train_acc'] = [min(1, max(0, v)) for v in history['train_acc']]
    history['val_acc'] = [min(1, max(0, v)) for v in history['val_acc']]

    save_path = "results/plots/demo_training.png" if save else None

    print("\nDisplaying simulated training curves...")

    plot_training_history(
        history,
        title="Training Progress (Simulated)",
        save_path=save_path
    )


def demo_benchmark_results(save: bool = False):
    """Demo 6: Benchmark results visualization."""
    print("\n" + "=" * 60)
    print("DEMO 6: Benchmark Results")
    print("=" * 60)

    # Simulate benchmark results
    np.random.seed(42)
    num_episodes = 100

    results = {
        'summary': {
            'success_rate': 0.87,
            'mean_length_ratio': 1.18,
            'median_length_ratio': 1.12,
            'num_episodes': num_episodes,
        },
        'episodes': [
            {
                'success': np.random.random() < 0.87,
                'length_ratio': 1.0 + np.random.exponential(0.15) if np.random.random() < 0.87 else None
            }
            for _ in range(num_episodes)
        ]
    }

    save_path = "results/plots/demo_benchmark.png" if save else None

    print(f"\nSuccess rate: {results['summary']['success_rate']:.1%}")
    print(f"Mean path ratio: {results['summary']['mean_length_ratio']:.2f}")
    print("Displaying benchmark results...")

    plot_benchmark_results(
        results,
        title="Benchmark Results (Simulated)",
        save_path=save_path
    )


def demo_animation(save: bool = False):
    """Demo 7: Navigation animation."""
    print("\n" + "=" * 60)
    print("DEMO 7: Navigation Animation")
    print("=" * 60)

    grid, start, goal = generate_solvable_map(size=8, seed=999)
    optimal_path = astar_path(grid, start, goal)

    save_path = "results/plots/demo_animation.gif" if save else None

    print(f"\nPath length: {len(optimal_path)} steps")
    print("Creating animation...")
    print("(This may take a moment)")

    if save:
        try:
            import PIL
            create_navigation_animation(
                grid, start, goal, optimal_path,
                save_path=save_path,
                interval=400
            )
            print(f"Animation saved to: {save_path}")
        except ImportError:
            print("Pillow not installed. Skipping animation save.")
            print("Install with: pip install pillow")
    else:
        print("Animation display requires interactive backend.")
        print("Use --save flag to save as GIF.")


def main():
    parser = argparse.ArgumentParser(description="NanoNav Visualization Demo")
    parser.add_argument("--save", action="store_true",
                        help="Save plots to results/plots/")
    parser.add_argument("--demo", type=int, default=0,
                        help="Run specific demo (1-7), 0 for all")
    args = parser.parse_args()

    print("=" * 60)
    print("NanoNav Visualization Demo")
    print("=" * 60)

    if args.save:
        print("\nPlots will be saved to results/plots/")
        Path("results/plots").mkdir(parents=True, exist_ok=True)

    demos = [
        (1, demo_ascii_grid, False),  # ASCII doesn't save
        (2, demo_grid_plot, True),
        (3, demo_action_sequence, True),
        (4, demo_path_comparison, True),
        (5, demo_training_history, True),
        (6, demo_benchmark_results, True),
        (7, demo_animation, True),
    ]

    for num, func, supports_save in demos:
        if args.demo == 0 or args.demo == num:
            if supports_save:
                func(save=args.save)
            else:
                func()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    if args.save:
        print("Plots saved to: results/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
