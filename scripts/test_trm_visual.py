#!/usr/bin/env python3
"""
Visual TRM Test

Run TRM navigation and compare to A*.

Usage:
    # Test untrained model
    python scripts/test_trm_visual.py

    # Test trained model
    python scripts/test_trm_visual.py --checkpoint checkpoints/best.pt

    # Custom options
    python scripts/test_trm_visual.py --seed 42 --size 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from trm_nav.map_generator import generate_solvable_map
from trm_nav.a_star import astar_path, path_to_actions, ACTION_NAMES, ACTIONS
from trm_nav.dataset import encode_state
from trm_nav.model import create_model
from trm_nav.visualize import plot_grid, plot_comparison, print_grid_ascii


def run_trm_rollout(model, grid, start, goal, max_steps=100, device='cpu'):
    """Run TRM step-by-step until goal or max steps."""
    model.eval()
    current = start
    path = [current]
    actions_taken = []

    with torch.no_grad():
        for step in range(max_steps):
            if current == goal:
                break

            # Encode state
            tokens = encode_state(grid, current, goal).unsqueeze(0).to(device)

            # Get prediction
            logits = model(tokens)
            action = logits.argmax(dim=-1).item()
            actions_taken.append(action)

            # Execute action
            dr, dc = ACTIONS[action]
            new_row = current[0] + dr
            new_col = current[1] + dc

            # Check validity
            rows, cols = grid.shape
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if grid[new_row, new_col] == 0:
                    current = (new_row, new_col)

            path.append(current)

            # Stuck detection (same position 3 times)
            if len(path) >= 3 and path[-1] == path[-2] == path[-3]:
                break

    success = current == goal
    return path, actions_taken, success


def main():
    parser = argparse.ArgumentParser(description="Visual TRM Test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--size", type=int, default=8, help="Grid size")
    parser.add_argument("--density", type=float, default=0.2, help="Obstacle density")
    parser.add_argument("--save", type=str, default=None, help="Save comparison image")
    parser.add_argument("--no-plot", action="store_true", help="ASCII only")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("TRM Navigation Test")
    print("=" * 50)
    print(f"Grid size: {args.size}x{args.size}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint or 'None (untrained)'}")
    print()

    # Generate map
    print("Generating map...")
    grid, start, goal = generate_solvable_map(
        size=args.size,
        obstacle_density=args.density,
        seed=args.seed
    )

    # Load model
    print("Loading model...")
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint.get('config', {})
        model = create_model(
            grid_size=config.get('grid_size', args.size),
            dim=config.get('dim', 64),
            depth=config.get('depth', 2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained model from {args.checkpoint}")
    else:
        model = create_model(grid_size=args.size)
        print("  Using untrained model (random predictions)")

    model = model.to(device)

    # Run A*
    print("\nRunning A*...")
    astar = astar_path(grid, start, goal)
    astar_actions = path_to_actions(astar)

    # Run TRM
    print("Running TRM...")
    trm_path, trm_actions, trm_success = run_trm_rollout(
        model, grid, start, goal, device=device
    )

    # Results
    print()
    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    print()
    print("A* (Optimal):")
    print(f"  Path length: {len(astar)} positions ({len(astar_actions)} moves)")
    print(f"  Actions: {' → '.join(ACTION_NAMES[a] for a in astar_actions)}")
    print()
    print("TRM (Learned):")
    print(f"  Path length: {len(trm_path)} positions ({len(trm_actions)} moves)")
    print(f"  Success: {'✓ Yes' if trm_success else '✗ No'}")
    if trm_actions:
        print(f"  Actions: {' → '.join(ACTION_NAMES[a] for a in trm_actions)}")

    # Metrics
    if trm_success:
        ratio = len(trm_path) / len(astar)
        print(f"\n  Path ratio: {ratio:.2f}x {'(optimal!)' if ratio == 1.0 else ''}")
    print()

    # ASCII visualization
    print("A* Path:")
    print_grid_ascii(grid, start, goal, astar)
    print()
    print("TRM Path:")
    print_grid_ascii(grid, start, goal, trm_path)

    # Matplotlib visualization
    if not args.no_plot:
        print()
        print("Displaying comparison... (close window to exit)")

        plot_comparison(
            grid, start, goal,
            trm_path, astar,
            title=f"TRM vs A* | Seed: {args.seed}",
            save_path=args.save,
            show=True
        )


if __name__ == "__main__":
    main()
