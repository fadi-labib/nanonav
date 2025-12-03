#!/usr/bin/env python3
"""
Run benchmark comparing TRM to A*.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --checkpoint checkpoints/best.pt --num-episodes 200
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from trm_nav.model import create_model
from trm_nav.evaluate import benchmark, print_benchmark_table, save_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRM Navigator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of test episodes")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Grid size for evaluation")
    parser.add_argument("--obstacle-density", type=float, default=0.2,
                        help="Obstacle density")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="results/benchmark.json",
                        help="Output path for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 50)
    print("NanoNav Benchmark")
    print("=" * 50)
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.num_episodes}")
    print(f"Grid size:  {args.grid_size}x{args.grid_size}")
    print("-" * 50)

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first: python -m trm_nav.train")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    model = create_model(
        grid_size=config.get('grid_size', args.grid_size),
        dim=config.get('dim', 64),
        depth=config.get('depth', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark(
        model,
        num_episodes=args.num_episodes,
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        base_seed=args.seed,
        device=device,
        verbose=True
    )

    # Print results
    print_benchmark_table(results['summary'])

    # Save results
    save_results(results, args.output)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
