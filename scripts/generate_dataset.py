#!/usr/bin/env python3
"""
Generate training and test datasets for TRM navigation.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --num-train 10000 --num-test 1000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trm_nav.dataset import build_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate TRM-Nav Dataset")
    parser.add_argument("--num-train", type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--num-test", type=int, default=500,
                        help="Number of test samples")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Grid size (8 for 8x8)")
    parser.add_argument("--obstacle-density", type=float, default=0.2,
                        help="Obstacle density (0.0 to 1.0)")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Random seed for training data")
    parser.add_argument("--test-seed", type=int, default=99999,
                        help="Random seed for test data")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for datasets")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (8x more data via rotations/flips)")

    args = parser.parse_args()

    print("=" * 50)
    print("TRM-Nav Dataset Generation")
    print("=" * 50)
    print(f"Grid size:        {args.grid_size}x{args.grid_size}")
    print(f"Obstacle density: {args.obstacle_density}")
    print(f"Training samples: {args.num_train}")
    print(f"Test samples:     {args.num_test}")
    print(f"Augmentation:     {'ON (8x via rotate/flip)' if args.augment else 'OFF'}")
    print("-" * 50)

    # Generate training data
    print("\nGenerating training data...")
    train_data = build_dataset(
        num_samples=args.num_train,
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        base_seed=args.train_seed,
        augment=args.augment
    )

    train_path = f"{args.output_dir}/train.pt"
    save_dataset(train_data, train_path)

    # Generate test data (never augment test data)
    print("\nGenerating test data...")
    test_data = build_dataset(
        num_samples=args.num_test,
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        base_seed=args.test_seed,
        augment=False  # Never augment test data
    )

    test_path = f"{args.output_dir}/test.pt"
    save_dataset(test_data, test_path)

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print("=" * 50)
    print(f"Training data: {train_path}")
    print(f"Test data:     {test_path}")
    print("\nNext step: python -m trm_nav.train")


if __name__ == "__main__":
    main()
