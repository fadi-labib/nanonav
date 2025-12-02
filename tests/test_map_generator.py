"""
Test 2: Map Generator

Run with:
    python tests/test_map_generator.py
    pytest tests/test_map_generator.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from trm_nav.map_generator import generate_random_map, generate_solvable_map, generate_dataset_maps
from trm_nav.a_star import astar_path


def test_random_map_shape():
    """Random maps have correct shape."""
    for size in [4, 8, 16]:
        grid = generate_random_map(size=size, seed=42)
        assert grid.shape == (size, size), f"Expected ({size}, {size}), got {grid.shape}"

    print("✓ Map shapes: 4x4, 8x8, 16x16")


def test_random_map_values():
    """Maps contain only 0s and 1s."""
    grid = generate_random_map(size=8, obstacle_density=0.3, seed=42)
    unique_vals = np.unique(grid)

    assert all(v in [0, 1] for v in unique_vals), f"Invalid values: {unique_vals}"
    print(f"✓ Map values: {unique_vals.tolist()} (only 0 and 1)")


def test_random_map_reproducibility():
    """Same seed produces same map."""
    grid1 = generate_random_map(size=8, seed=123)
    grid2 = generate_random_map(size=8, seed=123)

    assert np.array_equal(grid1, grid2), "Same seed should produce same map"
    print("✓ Reproducibility: same seed = same map")


def test_solvable_map_has_path():
    """Generated maps are actually solvable."""
    print("✓ Solvable maps (10 tests):")

    for i in range(10):
        grid, start, goal = generate_solvable_map(size=8, obstacle_density=0.2, seed=i)

        # Verify start and goal are free
        assert grid[start[0], start[1]] == 0, "Start should be free"
        assert grid[goal[0], goal[1]] == 0, "Goal should be free"

        # Verify path exists
        path = astar_path(grid, start, goal)
        assert path is not None, f"Map {i} should be solvable"
        assert len(path) >= 2, "Path should have at least 2 positions"

        if i < 3:  # Print first 3
            print(f"    Map {i}: start={start}, goal={goal}, path_len={len(path)}")


def test_solvable_map_different_positions():
    """Start and goal are different."""
    for i in range(10):
        _, start, goal = generate_solvable_map(size=8, seed=i + 100)
        assert start != goal, "Start and goal should be different"

    print("✓ Start != Goal in all maps")


def test_obstacle_density():
    """Obstacle density is approximately correct."""
    densities = []
    for i in range(20):
        grid = generate_random_map(size=8, obstacle_density=0.25, seed=i)
        actual_density = grid.sum() / (8 * 8)
        densities.append(actual_density)

    avg_density = np.mean(densities)
    assert 0.15 < avg_density < 0.35, f"Average density {avg_density} out of range"
    print(f"✓ Obstacle density: {avg_density:.2f} (target: 0.25)")


def test_generate_dataset_maps():
    """Batch map generation works."""
    maps = generate_dataset_maps(num_maps=20, size=8, base_seed=42)

    assert len(maps) > 0, "Should generate some maps"
    assert len(maps) <= 20, "Should not exceed requested count"

    for grid, start, goal in maps:
        assert grid.shape == (8, 8)
        path = astar_path(grid, start, goal)
        assert path is not None, "All maps should be solvable"

    print(f"✓ Batch generation: {len(maps)} solvable maps")


def run_all():
    """Run all tests with visual output."""
    print("\n" + "=" * 50)
    print("TEST 2: Map Generator")
    print("=" * 50 + "\n")

    test_random_map_shape()
    test_random_map_values()
    test_random_map_reproducibility()
    test_solvable_map_has_path()
    test_solvable_map_different_positions()
    test_obstacle_density()
    test_generate_dataset_maps()

    print("\n" + "-" * 50)
    print("All map generator tests passed!")
    print("-" * 50)


if __name__ == "__main__":
    run_all()
