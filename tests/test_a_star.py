"""
Test 1: A* Pathfinding Algorithm

Run with:
    python tests/test_a_star.py
    pytest tests/test_a_star.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from trm_nav.a_star import astar_path, path_to_actions, get_next_action, ACTION_NAMES


def test_simple_path():
    """A* finds path on empty grid."""
    grid = np.zeros((8, 8), dtype=np.int32)
    start = (0, 0)
    goal = (7, 7)

    path = astar_path(grid, start, goal)

    assert path is not None, "Path should exist"
    assert path[0] == start, "Path should start at start"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == 15, f"Optimal path length should be 15, got {len(path)}"

    print(f"✓ Empty grid: path length {len(path)}")


def test_path_with_obstacles():
    """A* navigates around obstacles."""
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)

    start = (0, 0)
    goal = (3, 5)

    path = astar_path(grid, start, goal)

    assert path is not None, "Path should exist"
    assert path[0] == start
    assert path[-1] == goal

    # Verify path avoids obstacles
    for pos in path:
        assert grid[pos[0], pos[1]] == 0, f"Path goes through obstacle at {pos}"

    print(f"✓ Obstacle avoidance: path length {len(path)}")
    print(f"  Path: {path}")


def test_no_path_exists():
    """A* returns None when blocked."""
    grid = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.int32)

    start = (0, 0)
    goal = (0, 2)

    path = astar_path(grid, start, goal)

    assert path is None, "No path should exist"
    print("✓ Blocked path: returns None")


def test_start_equals_goal():
    """A* handles start == goal."""
    grid = np.zeros((4, 4), dtype=np.int32)
    start = (2, 2)
    goal = (2, 2)

    path = astar_path(grid, start, goal)

    assert path is not None
    assert len(path) == 1
    assert path[0] == start
    print("✓ Start equals goal: path length 1")


def test_path_to_actions():
    """Convert path to action sequence."""
    path = [(0, 0), (1, 0), (1, 1), (2, 1)]
    actions = path_to_actions(path)

    assert len(actions) == 3
    assert actions[0] == 1, "First action should be DOWN"
    assert actions[1] == 3, "Second action should be RIGHT"
    assert actions[2] == 1, "Third action should be DOWN"

    action_names = [ACTION_NAMES[a] for a in actions]
    print(f"✓ Path to actions: {action_names}")


def test_get_next_action():
    """Get single next optimal action."""
    grid = np.zeros((4, 4), dtype=np.int32)

    # Moving right toward goal
    action = get_next_action(grid, (0, 0), (0, 3))
    assert action == 3, f"Should go RIGHT, got {ACTION_NAMES[action]}"

    # At goal
    action = get_next_action(grid, (2, 2), (2, 2))
    assert action == 4, f"Should STAY at goal, got {ACTION_NAMES[action]}"

    print("✓ Next action: RIGHT toward goal, STAY at goal")


def run_all():
    """Run all tests with visual output."""
    print("\n" + "=" * 50)
    print("TEST 1: A* Pathfinding")
    print("=" * 50 + "\n")

    test_simple_path()
    test_path_with_obstacles()
    test_no_path_exists()
    test_start_equals_goal()
    test_path_to_actions()
    test_get_next_action()

    print("\n" + "-" * 50)
    print("All A* tests passed!")
    print("-" * 50)


if __name__ == "__main__":
    run_all()
