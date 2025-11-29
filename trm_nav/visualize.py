"""
Visualization Utilities

Nice visual outputs for grids, paths, actions, and results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .a_star import ACTION_NAMES, ACTIONS


# Color scheme
COLORS = {
    'free': '#E8F5E9',       # Light green
    'obstacle': '#37474F',    # Dark gray
    'start': '#4CAF50',       # Green
    'goal': '#F44336',        # Red
    'path_astar': '#2196F3',  # Blue
    'path_trm': '#FF9800',    # Orange
    'current': '#9C27B0',     # Purple
    'visited': '#BBDEFB',     # Light blue
}


def plot_grid(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    astar_path: Optional[List[Tuple[int, int]]] = None,
    title: str = "Grid World",
    show_coords: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize a grid world with paths.

    Args:
        grid: 2D occupancy grid (0=free, 1=obstacle)
        start: Start position (row, col)
        goal: Goal position (row, col)
        path: Primary path to show (e.g., TRM path)
        astar_path: Secondary path for comparison (A* optimal)
        title: Plot title
        show_coords: Show coordinate labels
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    rows, cols = grid.shape

    # Create colored grid
    display = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                display[i, j] = plt.cm.colors.to_rgb(COLORS['obstacle'])
            else:
                display[i, j] = plt.cm.colors.to_rgb(COLORS['free'])

    ax.imshow(display, origin='upper', aspect='equal')

    # Draw grid lines
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Plot A* path (background)
    if astar_path and len(astar_path) > 1:
        astar_rows = [p[0] for p in astar_path]
        astar_cols = [p[1] for p in astar_path]
        ax.plot(astar_cols, astar_rows,
                color=COLORS['path_astar'], linewidth=3, alpha=0.5,
                marker='o', markersize=6, label=f'A* Path ({len(astar_path)-1} steps)')

    # Plot main path (foreground)
    if path and len(path) > 1:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows,
                color=COLORS['path_trm'], linewidth=4, alpha=0.8,
                marker='s', markersize=8, label=f'TRM Path ({len(path)-1} steps)')

    # Mark start
    if start:
        ax.plot(start[1], start[0], 'o', color=COLORS['start'],
                markersize=20, markeredgecolor='white', markeredgewidth=2)
        ax.annotate('S', (start[1], start[0]), ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

    # Mark goal
    if goal:
        ax.plot(goal[1], goal[0], '*', color=COLORS['goal'],
                markersize=25, markeredgecolor='white', markeredgewidth=1)
        ax.annotate('G', (goal[1], goal[0]), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    # Coordinate labels
    if show_coords:
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if path or astar_path:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_action_sequence(
    actions: List[int],
    title: str = "Action Sequence",
    figsize: Tuple[int, int] = (12, 2),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize a sequence of actions as colored boxes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    action_colors = {
        0: '#E3F2FD',  # UP - light blue
        1: '#FFF3E0',  # DOWN - light orange
        2: '#F3E5F5',  # LEFT - light purple
        3: '#E8F5E9',  # RIGHT - light green
        4: '#ECEFF1',  # STAY - gray
    }

    for i, action in enumerate(actions):
        color = action_colors.get(action, '#FFFFFF')
        rect = mpatches.FancyBboxPatch(
            (i, 0), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.45, ACTION_NAMES[action],
                ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(-0.2, len(actions) + 0.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_comparison(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    trm_path: List[Tuple[int, int]],
    astar_path: List[Tuple[int, int]],
    title: str = "Path Comparison: TRM vs A*",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Side-by-side comparison of TRM and A* paths.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    rows, cols = grid.shape

    for ax, path, path_color, name in [
        (axes[0], astar_path, COLORS['path_astar'], 'A* (Optimal)'),
        (axes[1], trm_path, COLORS['path_trm'], 'TRM (Learned)')
    ]:
        # Draw grid
        display = np.zeros((rows, cols, 3))
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 1:
                    display[i, j] = plt.cm.colors.to_rgb(COLORS['obstacle'])
                else:
                    display[i, j] = plt.cm.colors.to_rgb(COLORS['free'])

        ax.imshow(display, origin='upper')

        # Grid lines
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.5)

        # Draw path
        if path and len(path) > 1:
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            ax.plot(path_cols, path_rows, color=path_color, linewidth=3,
                    marker='o', markersize=8, alpha=0.8)

        # Start and goal
        ax.plot(start[1], start[0], 'o', color=COLORS['start'],
                markersize=18, markeredgecolor='white', markeredgewidth=2)
        ax.plot(goal[1], goal[0], '*', color=COLORS['goal'],
                markersize=22, markeredgecolor='white', markeredgewidth=1)

        # Labels
        success = path[-1] == goal if path else False
        status = "✓ Success" if success else "✗ Failed"
        path_len = len(path) - 1 if path else 0

        ax.set_title(f"{name}\n{path_len} steps | {status}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

    # Overall stats
    astar_len = len(astar_path) - 1 if astar_path else 0
    trm_len = len(trm_path) - 1 if trm_path else 0
    ratio = trm_len / astar_len if astar_len > 0 else float('inf')

    fig.suptitle(f"{title}\nPath Ratio: {ratio:.2f}x", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training loss and accuracy curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_benchmark_results(
    results: Dict,
    title: str = "Benchmark Results",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot benchmark results with bar charts.
    """
    summary = results.get('summary', results)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Success rate comparison
    agents = ['A*', 'TRM']
    success_rates = [100.0, summary.get('success_rate', 0) * 100]
    colors = [COLORS['path_astar'], COLORS['path_trm']]

    axes[0].bar(agents, success_rates, color=colors, edgecolor='black', linewidth=1)
    axes[0].axhline(85, color='green', linestyle='--', linewidth=2, label='Target (85%)')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate', fontsize=12)
    axes[0].set_ylim(0, 105)
    axes[0].legend()

    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Path length ratio distribution
    episodes = results.get('episodes', [])
    ratios = [e['length_ratio'] for e in episodes if e.get('length_ratio') is not None]

    if ratios:
        axes[1].hist(ratios, bins=20, color=COLORS['path_trm'],
                     edgecolor='black', alpha=0.7)
        axes[1].axvline(1.0, color='blue', linestyle='-', linewidth=2, label='Optimal (1.0)')
        axes[1].axvline(1.3, color='green', linestyle='--', linewidth=2, label='Target (≤1.3)')
        mean_ratio = np.mean(ratios)
        axes[1].axvline(mean_ratio, color='red', linestyle='-', linewidth=2,
                        label=f'Mean ({mean_ratio:.2f})')
        axes[1].set_xlabel('Path Length Ratio (TRM / A*)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Path Length Distribution', fontsize=12)
        axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def print_grid_ascii(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    Print grid as ASCII art (for terminal output).
    """
    rows, cols = grid.shape
    path_set = set(path) if path else set()

    print("┌" + "──" * cols + "┐")
    for i in range(rows):
        print("│", end="")
        for j in range(cols):
            pos = (i, j)
            if pos == start:
                print("🟢", end="")
            elif pos == goal:
                print("🎯", end="")
            elif pos in path_set:
                print("🟡", end="")
            elif grid[i, j] == 1:
                print("⬛", end="")
            else:
                print("⬜", end="")
        print("│")
    print("└" + "──" * cols + "┘")


def create_navigation_animation(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[Tuple[int, int]],
    save_path: Optional[str] = None,
    interval: int = 500
) -> animation.FuncAnimation:
    """
    Create animated visualization of navigation.

    Args:
        grid: Occupancy grid
        start: Start position
        goal: Goal position
        path: Path to animate
        save_path: Path to save as GIF
        interval: Milliseconds between frames

    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    rows, cols = grid.shape

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Draw grid
        display = np.zeros((rows, cols, 3))
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 1:
                    display[i, j] = plt.cm.colors.to_rgb(COLORS['obstacle'])
                else:
                    display[i, j] = plt.cm.colors.to_rgb(COLORS['free'])

        ax.imshow(display, origin='upper')

        # Grid lines
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.3)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.3)

        # Draw path up to current frame
        if frame > 0:
            partial_path = path[:frame + 1]
            path_rows = [p[0] for p in partial_path]
            path_cols = [p[1] for p in partial_path]
            ax.plot(path_cols, path_rows, color=COLORS['path_trm'],
                    linewidth=2, marker='o', markersize=4, alpha=0.5)

        # Current position
        current = path[frame]
        ax.plot(current[1], current[0], 'o', color=COLORS['current'],
                markersize=20, markeredgecolor='white', markeredgewidth=2)

        # Start and goal
        ax.plot(start[1], start[0], 's', color=COLORS['start'],
                markersize=12, markeredgecolor='white', markeredgewidth=1)
        ax.plot(goal[1], goal[0], '*', color=COLORS['goal'],
                markersize=18, markeredgecolor='white', markeredgewidth=1)

        ax.set_title(f"Step {frame + 1}/{len(path)}", fontsize=12)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        return []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(path), interval=interval, blit=False
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=1000 // interval)
        print(f"Saved animation: {save_path}")

    return anim


# Demo function
def demo():
    """Run visualization demo."""
    from .map_generator import generate_solvable_map
    from .a_star import astar_path, path_to_actions

    print("=" * 50)
    print("Visualization Demo")
    print("=" * 50)

    # Generate example
    grid, start, goal = generate_solvable_map(size=8, seed=42)
    optimal_path = astar_path(grid, start, goal)
    actions = path_to_actions(optimal_path)

    # ASCII visualization
    print("\n1. ASCII Grid:")
    print_grid_ascii(grid, start, goal, optimal_path)

    print(f"\nStart: {start}, Goal: {goal}")
    print(f"Path length: {len(optimal_path)} positions")
    print(f"Actions: {[ACTION_NAMES[a] for a in actions]}")

    # Matplotlib visualizations
    print("\n2. Grid Plot (close window to continue)...")
    plot_grid(grid, start, goal, path=optimal_path,
              title="Navigation Example")

    print("\n3. Action Sequence...")
    plot_action_sequence(actions, title="Action Sequence")

    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
