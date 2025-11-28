"""
Evaluation Module

Rollout simulation and benchmarking against A*.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .a_star import astar_path, ACTIONS, ACTION_NAMES
from .map_generator import generate_solvable_map
from .dataset import encode_state
from .model import create_model


def rollout_episode(
    model: torch.nn.Module,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 256,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Execute a single navigation episode with the TRM model.

    Args:
        model: Trained TRM model
        grid: Occupancy grid
        start: Start position
        goal: Goal position
        max_steps: Maximum steps before timeout
        device: Compute device

    Returns:
        Episode results dictionary
    """
    model.eval()

    current = start
    path = [current]
    actions_taken = []

    with torch.no_grad():
        for step in range(max_steps):
            if current == goal:
                break

            # Encode current state
            tokens = encode_state(grid, current, goal)
            tokens = tokens.unsqueeze(0).to(device)

            # Predict action
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

            # Check for stuck (same position twice in a row)
            if len(path) >= 2 and path[-1] == path[-2] and action != 4:
                # Tried to move but couldn't - might be stuck
                pass

    success = current == goal

    return {
        'success': success,
        'path': path,
        'actions': actions_taken,
        'steps': len(actions_taken),
        'path_length': len(path) - 1,  # Number of moves (not counting start)
        'reached': current,
        'goal': goal
    }


def compute_metrics(
    trm_result: Dict,
    astar_length: int
) -> Dict:
    """Compute evaluation metrics for a single episode."""
    metrics = {
        'success': trm_result['success'],
        'trm_length': trm_result['path_length'],
        'astar_length': astar_length,
        'steps': trm_result['steps']
    }

    if trm_result['success'] and astar_length > 0:
        metrics['length_ratio'] = trm_result['path_length'] / astar_length
    else:
        metrics['length_ratio'] = None

    return metrics


def benchmark(
    model: torch.nn.Module,
    num_episodes: int = 100,
    grid_size: int = 8,
    obstacle_density: float = 0.2,
    base_seed: int = 12345,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Dict:
    """
    Benchmark TRM against A* on random maps.

    Args:
        model: Trained TRM model
        num_episodes: Number of test episodes
        grid_size: Size of grids
        obstacle_density: Obstacle probability
        base_seed: Random seed for reproducibility
        device: Compute device
        verbose: Whether to show progress bar

    Returns:
        Benchmark results dictionary
    """
    model.eval()

    results = []
    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Benchmarking")

    for i in iterator:
        try:
            # Generate test map
            grid, start, goal = generate_solvable_map(
                size=grid_size,
                obstacle_density=obstacle_density,
                seed=base_seed + i
            )

            # Compute A* optimal path
            astar = astar_path(grid, start, goal)
            astar_length = len(astar) - 1 if astar else 0

            # Run TRM rollout
            max_steps = grid_size * grid_size * 4
            trm_result = rollout_episode(
                model, grid, start, goal,
                max_steps=max_steps,
                device=device
            )

            # Compute metrics
            metrics = compute_metrics(trm_result, astar_length)
            metrics['episode'] = i
            results.append(metrics)

        except ValueError:
            # Skip failed map generation
            continue

    # Aggregate results
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / len(results) if results else 0

    # Path length ratios (only for successful episodes)
    ratios = [r['length_ratio'] for r in results if r['length_ratio'] is not None]
    mean_ratio = np.mean(ratios) if ratios else float('inf')
    median_ratio = np.median(ratios) if ratios else float('inf')

    # Timeouts (exceeded max steps without reaching goal)
    timeouts = sum(1 for r in results if not r['success'])

    summary = {
        'num_episodes': len(results),
        'success_rate': success_rate,
        'mean_length_ratio': mean_ratio,
        'median_length_ratio': median_ratio,
        'timeouts': timeouts,
        'grid_size': grid_size
    }

    return {
        'summary': summary,
        'episodes': results
    }


def print_benchmark_table(summary: Dict) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 50)
    print("TRM-Nav PoC Benchmark Results")
    print("=" * 50)
    print(f"Grid Size: {summary['grid_size']}x{summary['grid_size']}")
    print(f"Episodes:  {summary['num_episodes']}")
    print("-" * 50)
    print(f"{'Agent':<12} {'Success':<12} {'Avg Ratio':<12} {'Timeouts':<12}")
    print("-" * 50)
    print(f"{'A*':<12} {'100.0%':<12} {'1.00':<12} {'0':<12}")
    print(f"{'TRM':<12} {summary['success_rate']*100:.1f}%{'':<6} "
          f"{summary['mean_length_ratio']:.2f}{'':<8} "
          f"{summary['timeouts']:<12}")
    print("=" * 50)

    # Success criteria check
    print("\nSuccess Criteria:")
    success_pass = summary['success_rate'] >= 0.85
    ratio_pass = summary['mean_length_ratio'] <= 1.3

    print(f"  Success Rate >= 85%: {'PASS' if success_pass else 'FAIL'} ({summary['success_rate']*100:.1f}%)")
    print(f"  Path Ratio <= 1.3:   {'PASS' if ratio_pass else 'FAIL'} ({summary['mean_length_ratio']:.2f})")
    print(f"\nOverall: {'PASS' if success_pass and ratio_pass else 'FAIL'}")


def save_results(results: Dict, path: str) -> None:
    """Save benchmark results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)

    print(f"Results saved to {path}")


def main():
    """Entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TRM Navigator")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Evaluating on {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    model = create_model(
        grid_size=config.get('grid_size', args.grid_size),
        dim=config.get('dim', 64),
        depth=config.get('depth', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Run benchmark
    results = benchmark(
        model,
        num_episodes=args.num_episodes,
        grid_size=args.grid_size,
        base_seed=args.seed,
        device=device
    )

    # Print and save results
    print_benchmark_table(results['summary'])
    save_results(results, args.output)


if __name__ == "__main__":
    main()
