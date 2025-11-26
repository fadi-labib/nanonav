"""
Dataset Module

Creates and manages navigation training data from A* demonstrations.
Includes data augmentation for improved generalization.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from .a_star import astar_path, path_to_actions
from .map_generator import generate_solvable_map


# Action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
# When we rotate/flip the grid, actions must be transformed accordingly

def rotate_action_90(action: int) -> int:
    """Rotate action 90° clockwise."""
    # UP→RIGHT, RIGHT→DOWN, DOWN→LEFT, LEFT→UP, STAY→STAY
    mapping = {0: 3, 1: 2, 2: 0, 3: 1, 4: 4}
    return mapping[action]


def rotate_action_180(action: int) -> int:
    """Rotate action 180°."""
    # UP→DOWN, DOWN→UP, LEFT→RIGHT, RIGHT→LEFT, STAY→STAY
    mapping = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}
    return mapping[action]


def rotate_action_270(action: int) -> int:
    """Rotate action 270° clockwise (= 90° counter-clockwise)."""
    # UP→LEFT, LEFT→DOWN, DOWN→RIGHT, RIGHT→UP, STAY→STAY
    mapping = {0: 2, 1: 3, 2: 1, 3: 0, 4: 4}
    return mapping[action]


def flip_action_horizontal(action: int) -> int:
    """Flip action horizontally (mirror left-right)."""
    # LEFT↔RIGHT, UP/DOWN/STAY unchanged
    mapping = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    return mapping[action]


def flip_action_vertical(action: int) -> int:
    """Flip action vertically (mirror up-down)."""
    # UP↔DOWN, LEFT/RIGHT/STAY unchanged
    mapping = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
    return mapping[action]


def rotate_position_90(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Rotate position 90° clockwise."""
    row, col = pos
    return (col, grid_size - 1 - row)


def rotate_position_180(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Rotate position 180°."""
    row, col = pos
    return (grid_size - 1 - row, grid_size - 1 - col)


def rotate_position_270(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Rotate position 270° clockwise."""
    row, col = pos
    return (grid_size - 1 - col, row)


def flip_position_horizontal(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Flip position horizontally."""
    row, col = pos
    return (row, grid_size - 1 - col)


def flip_position_vertical(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Flip position vertically."""
    row, col = pos
    return (grid_size - 1 - row, col)


def augment_sample(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    action: int
) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int]]:
    """
    Generate augmented versions of a single sample.

    Returns 8 versions:
        - Original
        - Rotate 90°, 180°, 270°
        - Flip horizontal, vertical
        - Flip both (= rotate 180° of flipped)

    Args:
        grid: Original grid
        start: Start position
        goal: Goal position
        action: Original action

    Returns:
        List of (grid, start, goal, action) tuples
    """
    grid_size = grid.shape[0]
    augmented = []

    # Original
    augmented.append((grid, start, goal, action))

    # Rotate 90°
    grid_90 = np.rot90(grid, k=-1)  # clockwise
    start_90 = rotate_position_90(start, grid_size)
    goal_90 = rotate_position_90(goal, grid_size)
    action_90 = rotate_action_90(action)
    augmented.append((grid_90, start_90, goal_90, action_90))

    # Rotate 180°
    grid_180 = np.rot90(grid, k=2)
    start_180 = rotate_position_180(start, grid_size)
    goal_180 = rotate_position_180(goal, grid_size)
    action_180 = rotate_action_180(action)
    augmented.append((grid_180, start_180, goal_180, action_180))

    # Rotate 270°
    grid_270 = np.rot90(grid, k=-3)  # = rot90 k=1 counter-clockwise
    start_270 = rotate_position_270(start, grid_size)
    goal_270 = rotate_position_270(goal, grid_size)
    action_270 = rotate_action_270(action)
    augmented.append((grid_270, start_270, goal_270, action_270))

    # Flip horizontal
    grid_fh = np.fliplr(grid)
    start_fh = flip_position_horizontal(start, grid_size)
    goal_fh = flip_position_horizontal(goal, grid_size)
    action_fh = flip_action_horizontal(action)
    augmented.append((grid_fh, start_fh, goal_fh, action_fh))

    # Flip vertical
    grid_fv = np.flipud(grid)
    start_fv = flip_position_vertical(start, grid_size)
    goal_fv = flip_position_vertical(goal, grid_size)
    action_fv = flip_action_vertical(action)
    augmented.append((grid_fv, start_fv, goal_fv, action_fv))

    # Flip horizontal + rotate 90° (diagonal flip)
    grid_fh90 = np.rot90(np.fliplr(grid), k=-1)
    start_fh90 = rotate_position_90(flip_position_horizontal(start, grid_size), grid_size)
    goal_fh90 = rotate_position_90(flip_position_horizontal(goal, grid_size), grid_size)
    action_fh90 = rotate_action_90(flip_action_horizontal(action))
    augmented.append((grid_fh90, start_fh90, goal_fh90, action_fh90))

    # Flip vertical + rotate 90° (other diagonal flip)
    grid_fv90 = np.rot90(np.flipud(grid), k=-1)
    start_fv90 = rotate_position_90(flip_position_vertical(start, grid_size), grid_size)
    goal_fv90 = rotate_position_90(flip_position_vertical(goal, grid_size), grid_size)
    action_fv90 = rotate_action_90(flip_action_vertical(action))
    augmented.append((grid_fv90, start_fv90, goal_fv90, action_fv90))

    return augmented


class NavigationDataset(Dataset):
    """
    PyTorch Dataset for navigation training.

    Each sample contains:
        - tokens: Encoded state (flattened grid + start/goal coords)
        - action: Target action ID (0-4)
    """

    def __init__(self, data_path: Optional[str] = None, data: Optional[dict] = None):
        """
        Initialize dataset from file or in-memory data.

        Args:
            data_path: Path to .pt file containing dataset
            data: Dictionary with 'tokens' and 'actions' tensors
        """
        if data_path is not None:
            loaded = torch.load(data_path)
            self.tokens = loaded['tokens']
            self.actions = loaded['actions']
        elif data is not None:
            self.tokens = data['tokens']
            self.actions = data['actions']
        else:
            raise ValueError("Must provide either data_path or data")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.actions[idx]


def encode_state(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> torch.Tensor:
    """
    Encode navigation state as token sequence.

    Encoding:
        - Free cell: 1
        - Obstacle: 2
        - Coordinates: raw integers + 3 (to avoid overlap with 0,1,2)

    Args:
        grid: 2D occupancy grid
        start: Start position (row, col)
        goal: Goal position (row, col)

    Returns:
        1D tensor of token IDs
    """
    # Flatten grid and encode (0 -> 1, 1 -> 2)
    flat_grid = grid.flatten() + 1

    # Encode coordinates (offset by 3 to avoid collision)
    coords = np.array([
        int(start[0]) + 3,
        int(start[1]) + 3,
        int(goal[0]) + 3,
        int(goal[1]) + 3
    ])

    # Concatenate
    tokens = np.concatenate([flat_grid, coords])

    return torch.tensor(tokens, dtype=torch.long)


def build_dataset(
    num_samples: int,
    grid_size: int = 8,
    obstacle_density: float = 0.2,
    base_seed: int = 42,
    augment: bool = False
) -> dict:
    """
    Build training dataset from A* demonstrations.

    For each sample:
        1. Generate random solvable map
        2. Compute A* path
        3. For each step along path, create (state, action) pair
        4. Optionally augment with rotations/flips (8x data)

    Args:
        num_samples: Target number of (state, action) pairs
        grid_size: Size of grid
        obstacle_density: Obstacle probability
        base_seed: Random seed for reproducibility
        augment: Whether to apply data augmentation (8x samples)

    Returns:
        Dictionary with 'tokens' and 'actions' tensors
    """
    all_tokens = []
    all_actions = []

    # If augmenting, we need fewer base samples
    target = num_samples // 8 if augment else num_samples

    map_idx = 0

    while len(all_actions) < target:
        try:
            # Generate solvable map
            grid, start, goal = generate_solvable_map(
                size=grid_size,
                obstacle_density=obstacle_density,
                seed=base_seed + map_idx
            )

            # Get optimal path
            path = astar_path(grid, start, goal)

            if path is None or len(path) < 2:
                map_idx += 1
                continue

            # Get actions for this path
            actions = path_to_actions(path)

            # Create training sample for each step
            for step_idx, action in enumerate(actions):
                current_pos = path[step_idx]

                if augment:
                    # Generate 8 augmented versions
                    aug_samples = augment_sample(grid, current_pos, goal, action)
                    for aug_grid, aug_start, aug_goal, aug_action in aug_samples:
                        tokens = encode_state(aug_grid, aug_start, aug_goal)
                        all_tokens.append(tokens)
                        all_actions.append(aug_action)
                else:
                    # Just the original
                    tokens = encode_state(grid, current_pos, goal)
                    all_tokens.append(tokens)
                    all_actions.append(action)

                if len(all_actions) >= num_samples:
                    break

            map_idx += 1

        except ValueError:
            map_idx += 1
            continue

    # Trim to exact size
    all_tokens = all_tokens[:num_samples]
    all_actions = all_actions[:num_samples]

    return {
        'tokens': torch.stack(all_tokens),
        'actions': torch.tensor(all_actions, dtype=torch.long)
    }


def save_dataset(data: dict, path: str) -> None:
    """Save dataset to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"Saved dataset with {len(data['actions'])} samples to {path}")


def load_dataset(path: str) -> dict:
    """Load dataset from disk."""
    return torch.load(path)
