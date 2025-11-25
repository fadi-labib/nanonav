"""
Map Generator

Generates random occupancy grids with guaranteed solvability.
"""

import numpy as np
from typing import Tuple, Optional
from .a_star import astar_path


def generate_random_map(
    size: int = 8,
    obstacle_density: float = 0.2,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random occupancy grid.

    Args:
        size: Grid size (size x size)
        obstacle_density: Probability of each cell being an obstacle
        seed: Random seed for reproducibility

    Returns:
        2D numpy array (0=free, 1=obstacle)
    """
    if seed is not None:
        np.random.seed(seed)

    grid = (np.random.random((size, size)) < obstacle_density).astype(np.int32)
    return grid


def generate_solvable_map(
    size: int = 8,
    obstacle_density: float = 0.2,
    seed: Optional[int] = None,
    max_attempts: int = 100
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Generate a random map with guaranteed solvable start/goal.

    Args:
        size: Grid size
        obstacle_density: Obstacle probability
        seed: Random seed
        max_attempts: Maximum attempts to find solvable configuration

    Returns:
        Tuple of (grid, start_pos, goal_pos)

    Raises:
        ValueError: If no solvable configuration found
    """
    rng = np.random.RandomState(seed)

    for attempt in range(max_attempts):
        # Generate random grid
        grid = (rng.random((size, size)) < obstacle_density).astype(np.int32)

        # Find free cells
        free_cells = list(zip(*np.where(grid == 0)))

        if len(free_cells) < 2:
            continue

        # Randomly select start and goal from free cells
        indices = rng.choice(len(free_cells), size=2, replace=False)
        start = free_cells[indices[0]]
        goal = free_cells[indices[1]]

        # Check if path exists
        path = astar_path(grid, start, goal)

        if path is not None and len(path) > 1:
            return grid, start, goal

    raise ValueError(f"Could not generate solvable map after {max_attempts} attempts")


def generate_dataset_maps(
    num_maps: int,
    size: int = 8,
    obstacle_density: float = 0.2,
    base_seed: int = 42
) -> list:
    """
    Generate multiple solvable maps for dataset creation.

    Args:
        num_maps: Number of maps to generate
        size: Grid size
        obstacle_density: Obstacle probability
        base_seed: Base random seed

    Returns:
        List of (grid, start, goal) tuples
    """
    maps = []

    for i in range(num_maps):
        try:
            grid, start, goal = generate_solvable_map(
                size=size,
                obstacle_density=obstacle_density,
                seed=base_seed + i
            )
            maps.append((grid, start, goal))
        except ValueError:
            # Skip failed attempts
            continue

    return maps
