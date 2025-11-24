"""
A* Pathfinding Algorithm

Provides optimal path computation for grid-based navigation.
Used as the teacher/oracle for training the TRM model.
"""

import heapq
from typing import List, Tuple, Optional
import numpy as np


# Action definitions
# 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
ACTIONS = {
    0: (-1, 0),   # UP (decrease row)
    1: (1, 0),    # DOWN (increase row)
    2: (0, -1),   # LEFT (decrease col)
    3: (0, 1),    # RIGHT (increase col)
    4: (0, 0),    # STAY
}

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(
    pos: Tuple[int, int],
    grid: np.ndarray
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Get valid neighboring positions and the action to reach them.

    Args:
        pos: Current position (row, col)
        grid: Occupancy grid (0=free, 1=obstacle)

    Returns:
        List of (neighbor_pos, action_id) tuples
    """
    neighbors = []
    rows, cols = grid.shape

    for action_id, (dr, dc) in ACTIONS.items():
        if action_id == 4:  # Skip STAY for pathfinding
            continue

        new_row = pos[0] + dr
        new_col = pos[1] + dc

        # Check bounds
        if 0 <= new_row < rows and 0 <= new_col < cols:
            # Check if free
            if grid[new_row, new_col] == 0:
                neighbors.append(((new_row, new_col), action_id))

    return neighbors


def astar_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    Compute the optimal path using A* algorithm.

    Args:
        grid: 2D occupancy grid (0=free, 1=obstacle)
        start: Start position (row, col)
        goal: Goal position (row, col)

    Returns:
        List of positions from start to goal (inclusive), or None if no path exists
    """
    # Validate inputs
    if grid[start[0], start[1]] == 1:
        return None  # Start is blocked
    if grid[goal[0], goal[1]] == 1:
        return None  # Goal is blocked

    if start == goal:
        return [start]

    # Priority queue: (f_score, counter, position)
    # counter is used to break ties
    counter = 0
    open_set = [(heuristic(start, goal), counter, start)]

    # Track where we came from
    came_from = {}

    # g_score: cost from start to current node
    g_score = {start: 0}

    # Set of visited nodes
    closed_set = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        closed_set.add(current)

        for neighbor, _ in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return None  # No path found


def path_to_actions(path: List[Tuple[int, int]]) -> List[int]:
    """
    Convert a path (list of positions) to a list of action IDs.

    Args:
        path: List of (row, col) positions

    Returns:
        List of action IDs to move along the path
    """
    if len(path) < 2:
        return []

    actions = []
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]

        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]

        # Find matching action
        for action_id, (adr, adc) in ACTIONS.items():
            if dr == adr and dc == adc:
                actions.append(action_id)
                break

    return actions


def get_next_action(
    grid: np.ndarray,
    current: Tuple[int, int],
    goal: Tuple[int, int]
) -> int:
    """
    Get the optimal next action from current position toward goal.

    Args:
        grid: Occupancy grid
        current: Current position
        goal: Goal position

    Returns:
        Action ID (0-4), returns STAY (4) if at goal or no path
    """
    if current == goal:
        return 4  # STAY

    path = astar_path(grid, current, goal)

    if path is None or len(path) < 2:
        return 4  # No path, stay

    actions = path_to_actions(path)
    return actions[0] if actions else 4
