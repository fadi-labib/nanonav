"""
Test 3: Dataset Creation

Run with:
    python tests/test_dataset.py
    pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from trm_nav.dataset import encode_state, build_dataset, NavigationDataset
from trm_nav.a_star import ACTION_NAMES


def test_encode_state_length():
    """Encoded state has correct length."""
    grid = np.zeros((8, 8), dtype=np.int32)
    tokens = encode_state(grid, (0, 0), (7, 7))

    expected_len = 8 * 8 + 4  # 64 grid + 4 coords
    assert len(tokens) == expected_len, f"Expected {expected_len}, got {len(tokens)}"
    print(f"✓ Token length: {len(tokens)} (64 grid + 4 coords)")


def test_encode_state_values():
    """Encoding uses correct token values."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[2, 3] = 1  # Add obstacle

    tokens = encode_state(grid, (0, 0), (7, 7))

    # Free cells should be 1
    assert tokens[0] == 1, "Free cell should encode to 1"

    # Obstacle should be 2
    obstacle_idx = 2 * 8 + 3  # row 2, col 3
    assert tokens[obstacle_idx] == 2, f"Obstacle should encode to 2, got {tokens[obstacle_idx]}"

    print(f"✓ Token values: free=1, obstacle=2")


def test_encode_state_coords():
    """Coordinates are appended correctly."""
    grid = np.zeros((8, 8), dtype=np.int32)
    start = (1, 2)
    goal = (5, 6)

    tokens = encode_state(grid, start, goal)

    # Last 4 tokens are coords (offset by 3)
    coords = tokens[-4:].tolist()
    expected = [start[0] + 3, start[1] + 3, goal[0] + 3, goal[1] + 3]

    assert coords == expected, f"Expected {expected}, got {coords}"
    print(f"✓ Coords encoded: start={start}, goal={goal}")
    print(f"  Last 4 tokens: {coords}")


def test_build_dataset_shapes():
    """Built dataset has correct shapes."""
    data = build_dataset(num_samples=100, grid_size=8, base_seed=42)

    assert 'tokens' in data, "Dataset should have 'tokens'"
    assert 'actions' in data, "Dataset should have 'actions'"

    assert data['tokens'].shape[0] == 100, "Should have 100 samples"
    assert data['tokens'].shape[1] == 68, "Token length should be 68"
    assert data['actions'].shape[0] == 100, "Should have 100 actions"

    print(f"✓ Dataset shapes: tokens={tuple(data['tokens'].shape)}, actions={tuple(data['actions'].shape)}")


def test_build_dataset_action_range():
    """Actions are in valid range [0, 4]."""
    data = build_dataset(num_samples=100, grid_size=8, base_seed=42)

    min_action = data['actions'].min().item()
    max_action = data['actions'].max().item()

    assert min_action >= 0, f"Min action {min_action} < 0"
    assert max_action <= 4, f"Max action {max_action} > 4"

    print(f"✓ Action range: [{min_action}, {max_action}] (valid: 0-4)")


def test_build_dataset_action_distribution():
    """Actions have reasonable distribution."""
    data = build_dataset(num_samples=200, grid_size=8, base_seed=42)

    print("✓ Action distribution:")
    for i in range(5):
        count = (data['actions'] == i).sum().item()
        pct = 100 * count / len(data['actions'])
        print(f"    {ACTION_NAMES[i]}: {count} ({pct:.1f}%)")

    # At least 3 different actions should be present
    unique_actions = data['actions'].unique()
    assert len(unique_actions) >= 3, "Should have at least 3 different actions"


def test_navigation_dataset_class():
    """NavigationDataset class works correctly."""
    data = build_dataset(num_samples=50, grid_size=8, base_seed=99)
    dataset = NavigationDataset(data=data)

    assert len(dataset) == 50, f"Dataset length should be 50, got {len(dataset)}"

    tokens, action = dataset[0]
    assert tokens.shape == (68,), f"Token shape should be (68,), got {tokens.shape}"
    assert isinstance(action.item(), int), "Action should be integer"

    print(f"✓ NavigationDataset: len={len(dataset)}, sample shape={tuple(tokens.shape)}")


def test_dataset_reproducibility():
    """Same seed produces same dataset."""
    data1 = build_dataset(num_samples=50, grid_size=8, base_seed=123)
    data2 = build_dataset(num_samples=50, grid_size=8, base_seed=123)

    assert torch.equal(data1['tokens'], data2['tokens']), "Tokens should match"
    assert torch.equal(data1['actions'], data2['actions']), "Actions should match"

    print("✓ Reproducibility: same seed = same dataset")


def run_all():
    """Run all tests with visual output."""
    print("\n" + "=" * 50)
    print("TEST 3: Dataset Creation")
    print("=" * 50 + "\n")

    test_encode_state_length()
    test_encode_state_values()
    test_encode_state_coords()
    test_build_dataset_shapes()
    test_build_dataset_action_range()
    test_build_dataset_action_distribution()
    test_navigation_dataset_class()
    test_dataset_reproducibility()

    print("\n" + "-" * 50)
    print("All dataset tests passed!")
    print("-" * 50)


if __name__ == "__main__":
    run_all()
