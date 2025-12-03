# Implementation Notes

This document contains detailed technical notes on the NanoNav implementation, including design decisions, lessons learned, and code patterns that can be reused in other projects.

## Table of Contents

1. [A* Pathfinding](#a-pathfinding)
2. [Map Generation](#map-generation)
3. [State Encoding](#state-encoding)
4. [Data Augmentation](#data-augmentation)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Regularization Techniques](#regularization-techniques)
8. [Lessons Learned](#lessons-learned)

---

## A* Pathfinding

### Implementation (`a_star.py`)

Uses NetworkX for graph representation and built-in A* algorithm:

```python
import networkx as nx

def astar_path(grid: np.ndarray, start: Tuple, goal: Tuple) -> Optional[List[Tuple]]:
    """Find optimal path using A* algorithm."""
    G = nx.grid_2d_graph(rows, cols)

    # Remove obstacle nodes
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:  # Obstacle
                G.remove_node((r, c))

    # Manhattan distance heuristic (admissible for grid movement)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    return nx.astar_path(G, start, goal, heuristic=heuristic)
```

### Action Mapping

```python
# Action IDs and their effects
ACTIONS = {
    0: (-1, 0),   # UP: row decreases
    1: (1, 0),    # DOWN: row increases
    2: (0, -1),   # LEFT: col decreases
    3: (0, 1),    # RIGHT: col increases
    4: (0, 0),    # STAY: no movement
}

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
```

### Path to Actions Conversion

```python
def path_to_actions(path: List[Tuple]) -> List[int]:
    """Convert path coordinates to action sequence."""
    actions = []
    for i in range(len(path) - 1):
        curr, next_pos = path[i], path[i + 1]
        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]

        for action_id, (ar, ac) in ACTIONS.items():
            if (dr, dc) == (ar, ac):
                actions.append(action_id)
                break
    return actions
```

---

## Map Generation

### Solvable Map Generation (`map_generator.py`)

Key insight: Generate maps with guaranteed solvability by validating with A* before returning.

```python
def generate_solvable_map(size: int, obstacle_density: float, seed: int):
    """Generate a random map that is guaranteed to have a valid path."""
    rng = np.random.default_rng(seed)
    max_attempts = 100

    for attempt in range(max_attempts):
        # Create empty grid
        grid = np.zeros((size, size), dtype=np.int32)

        # Add random obstacles
        num_obstacles = int(size * size * obstacle_density)
        for _ in range(num_obstacles):
            r, c = rng.integers(0, size), rng.integers(0, size)
            grid[r, c] = 1

        # Random start and goal (ensure they're free)
        start = tuple(rng.integers(0, size, 2))
        goal = tuple(rng.integers(0, size, 2))
        grid[start[0], start[1]] = 0
        grid[goal[0], goal[1]] = 0

        # Verify solvability
        if start != goal and astar_path(grid, start, goal) is not None:
            return grid, start, goal

    raise ValueError(f"Could not generate solvable map after {max_attempts} attempts")
```

### Design Decisions

1. **Deterministic seeding**: Each map has a unique seed for reproducibility
2. **Density control**: Typical values 0.15-0.25 for reasonable difficulty
3. **Validation**: Never return unsolvable maps (wastes training time)

---

## State Encoding

### Token-Based Encoding (`dataset.py`)

The state is encoded as a 1D sequence of tokens for the Transformer/Mixer architecture:

```python
def encode_state(grid: np.ndarray, start: Tuple, goal: Tuple) -> torch.Tensor:
    """
    Encode navigation state as token sequence.

    Token values:
        - 1: Free cell (grid value 0)
        - 2: Obstacle (grid value 1)
        - 3+: Coordinate values (offset to avoid collision)

    Returns: [grid_tokens..., start_row+3, start_col+3, goal_row+3, goal_col+3]
    """
    # Flatten grid: 0 -> 1 (free), 1 -> 2 (obstacle)
    flat_grid = grid.flatten() + 1

    # Encode coordinates with offset
    coords = np.array([
        int(start[0]) + 3,
        int(start[1]) + 3,
        int(goal[0]) + 3,
        int(goal[1]) + 3
    ])

    tokens = np.concatenate([flat_grid, coords])
    return torch.tensor(tokens, dtype=torch.long)
```

### Sequence Length Calculation

```
seq_len = grid_size × grid_size + 4
        = 8 × 8 + 4
        = 68 tokens (for 8×8 grid)
```

### Why Token-Based?

1. **Compatibility**: Works with Transformer/Mixer architectures
2. **Simplicity**: No special 2D convolutions needed
3. **Flexibility**: Easy to extend with additional features

### Trade-offs

- **Pro**: Simple, works with any sequence model
- **Con**: Loses 2D spatial structure (though Mixer can learn it)

---

## Data Augmentation

### Overview

Data augmentation multiplies training data by 8× through geometric transformations. Critical insight: **actions must be transformed consistently with the grid**.

### Transformation Functions

#### Grid Transformations

```python
# Rotations
grid_90 = np.rot90(grid, k=-1)   # 90° clockwise
grid_180 = np.rot90(grid, k=2)   # 180°
grid_270 = np.rot90(grid, k=-3)  # 270° clockwise (= 90° counter-clockwise)

# Flips
grid_fh = np.fliplr(grid)  # Horizontal flip (mirror left-right)
grid_fv = np.flipud(grid)  # Vertical flip (mirror up-down)
```

#### Position Transformations

```python
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
```

#### Action Transformations (Critical!)

```python
def rotate_action_90(action: int) -> int:
    """Rotate action 90° clockwise.

    When grid rotates 90° CW:
    - What was UP (going to smaller row) now goes RIGHT (larger col)
    - What was RIGHT (going to larger col) now goes DOWN (larger row)
    - What was DOWN (going to larger row) now goes LEFT (smaller col)
    - What was LEFT (going to smaller col) now goes UP (smaller row)
    """
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
```

### Complete Augmentation Function

```python
def augment_sample(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    action: int
) -> List[Tuple[np.ndarray, Tuple, Tuple, int]]:
    """Generate 8 augmented versions of a single sample."""
    grid_size = grid.shape[0]
    augmented = []

    # 1. Original
    augmented.append((grid, start, goal, action))

    # 2. Rotate 90°
    grid_90 = np.rot90(grid, k=-1)
    start_90 = rotate_position_90(start, grid_size)
    goal_90 = rotate_position_90(goal, grid_size)
    action_90 = rotate_action_90(action)
    augmented.append((grid_90, start_90, goal_90, action_90))

    # 3. Rotate 180°
    grid_180 = np.rot90(grid, k=2)
    start_180 = rotate_position_180(start, grid_size)
    goal_180 = rotate_position_180(goal, grid_size)
    action_180 = rotate_action_180(action)
    augmented.append((grid_180, start_180, goal_180, action_180))

    # 4. Rotate 270°
    grid_270 = np.rot90(grid, k=-3)
    start_270 = rotate_position_270(start, grid_size)
    goal_270 = rotate_position_270(goal, grid_size)
    action_270 = rotate_action_270(action)
    augmented.append((grid_270, start_270, goal_270, action_270))

    # 5. Flip horizontal
    grid_fh = np.fliplr(grid)
    start_fh = flip_position_horizontal(start, grid_size)
    goal_fh = flip_position_horizontal(goal, grid_size)
    action_fh = flip_action_horizontal(action)
    augmented.append((grid_fh, start_fh, goal_fh, action_fh))

    # 6. Flip vertical
    grid_fv = np.flipud(grid)
    start_fv = flip_position_vertical(start, grid_size)
    goal_fv = flip_position_vertical(goal, grid_size)
    action_fv = flip_action_vertical(action)
    augmented.append((grid_fv, start_fv, goal_fv, action_fv))

    # 7. Flip horizontal + rotate 90° (diagonal flip)
    grid_fh90 = np.rot90(np.fliplr(grid), k=-1)
    start_fh90 = rotate_position_90(flip_position_horizontal(start, grid_size), grid_size)
    goal_fh90 = rotate_position_90(flip_position_horizontal(goal, grid_size), grid_size)
    action_fh90 = rotate_action_90(flip_action_horizontal(action))
    augmented.append((grid_fh90, start_fh90, goal_fh90, action_fh90))

    # 8. Flip vertical + rotate 90° (other diagonal flip)
    grid_fv90 = np.rot90(np.flipud(grid), k=-1)
    start_fv90 = rotate_position_90(flip_position_vertical(start, grid_size), grid_size)
    goal_fv90 = rotate_position_90(flip_position_vertical(goal, grid_size), grid_size)
    action_fv90 = rotate_action_90(flip_action_vertical(action))
    augmented.append((grid_fv90, start_fv90, goal_fv90, action_fv90))

    return augmented
```

### Why These 8 Transformations?

These 8 transformations form the **dihedral group D4** - all symmetries of a square:
- 4 rotations (0°, 90°, 180°, 270°)
- 4 reflections (horizontal, vertical, 2 diagonals)

This teaches the model that navigation principles are orientation-invariant.

---

## Model Architecture

### TRM Wrapper (`model.py`)

The `tiny-recursive-model` library is designed for sequence-to-sequence tasks. For classification, we adapted it:

```python
class TRMNavigator(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_tokens: int = 256,
        seq_len: int = 68,
        depth: int = 2,
        num_actions: int = 5,
        max_recursion_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if HAS_TRM:
            self.trm = TinyRecursiveModel(
                dim=dim,
                num_tokens=num_tokens,
                network=MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
            )
        else:
            self.trm = FallbackTRM(...)  # Simple MLP fallback

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_actions)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_trm:
            # Use internal components for classification
            embedded = self.trm.input_embed(tokens)  # (batch, seq_len, dim)

            features = embedded
            for _ in range(self.max_recursion_steps):
                features = self.trm.network(features)

            # Pool over sequence
            features = features.mean(dim=1)  # (batch, dim)
            features = self.feature_dropout(features)
        else:
            features = self.trm(tokens)

        return self.classifier(features)
```

### Key Design Decisions

1. **Mean pooling**: Average over sequence dimension for fixed-size output
2. **Recursive refinement**: Apply network N times (default 8)
3. **Dropout placement**: After pooling and in classifier head
4. **Fallback mode**: Works without TRM library installed

### Parameter Counts

| Configuration | Parameters |
|---------------|------------|
| dim=64, depth=2 | ~50,000 |
| dim=128, depth=3 | ~200,000 |
| dim=256, depth=4 | ~800,000 |

---

## Training Pipeline

### Training Loop (`train.py`)

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for tokens, actions in dataloader:
        tokens, actions = tokens.to(device), actions.to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, actions)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * tokens.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total += tokens.size(0)

    return {'loss': total_loss / total, 'accuracy': correct / total}
```

### Optimizer Configuration

```python
# AdamW: Adam with decoupled weight decay (better regularization)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Cosine annealing: Smooth learning rate decay
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Cross-entropy for multi-class classification
criterion = nn.CrossEntropyLoss()
```

---

## Regularization Techniques

### 1. Dropout

Randomly zeros neuron outputs during training (prevents co-adaptation):

```python
# In model
self.classifier = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Dropout(dropout),  # After layer norm
    nn.Linear(dim, dim),
    nn.GELU(),
    nn.Dropout(dropout),  # Before final layer
    nn.Linear(dim, num_actions)
)
```

**Typical values**: 0.1 - 0.3

### 2. Weight Decay (L2 Regularization)

Penalizes large weights, encouraging simpler solutions:

```python
# AdamW applies weight decay correctly (not to bias terms)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
```

**Typical values**: 0.01 - 0.1

### 3. Early Stopping

Stops training when validation loss stops improving:

```python
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop
```

**Typical patience**: 10-20 epochs

### 4. Gradient Clipping

Prevents exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Lessons Learned

### 1. Overfitting is the Main Challenge

**Symptom**: Train accuracy 100%, validation accuracy ~70%

**Root cause**: Small dataset + expressive model = memorization

**Solutions** (in order of effectiveness):
1. More data (50k+ samples)
2. Data augmentation (8× via rotations/flips)
3. Dropout (0.1-0.2)
4. Weight decay (0.01)
5. Early stopping

### 2. Library API Mismatch

The `tiny-recursive-model` library was designed for sequence-to-sequence generation, not classification.

**Solution**: Use internal components directly:
```python
# Don't use: self.trm(tokens)  # Expects additional args
# Do use:
embedded = self.trm.input_embed(tokens)
features = self.trm.network(embedded)
```

### 3. Action Transformation is Critical

When augmenting data with rotations/flips, the action labels MUST be transformed consistently.

**Wrong**: Only transform grid, keep original action
**Right**: Transform grid, position, AND action together

### 4. Validation Loss > Training Loss is Normal

Due to dropout being disabled during evaluation, validation loss is typically slightly higher even without overfitting.

**Concern**: When val_loss is 10x+ higher than train_loss, or when val_loss increases while train_loss decreases.

### 5. Save Best Model by Validation Loss

Don't save the model with highest validation accuracy - save the one with lowest validation loss. Loss is more informative about confidence.

```python
if val_loss < best_val_loss:  # Not: if val_acc > best_val_acc
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best.pt")
```

### 6. Test on Diverse Seeds

A model might succeed on specific seeds but fail on others. Always test on multiple random seeds (not just training seeds).

---

## Reusable Patterns

### Pattern: Behavioral Cloning Pipeline

```
1. Expert (A*) generates demonstrations
2. Convert to (state, action) pairs
3. Train classifier to predict actions
4. Evaluate via rollouts
```

### Pattern: Geometric Data Augmentation

```
1. Define grid transformation (rotate, flip)
2. Define position transformation (same geometric op)
3. Define action transformation (direction changes)
4. Apply all three consistently to each sample
```

### Pattern: Early Stopping with Best Model

```python
best_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_checkpoint(model, "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### Pattern: Fallback Implementation

```python
try:
    from external_library import Model
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False

class MyModel(nn.Module):
    def __init__(self):
        if HAS_LIBRARY:
            self.model = Model(...)
        else:
            self.model = SimpleFallback(...)
```

---

## References

- [MLP-Mixer Paper](https://arxiv.org/abs/2105.01601)
- [AdamW Paper](https://arxiv.org/abs/1711.05101)
- [A* Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Behavioral Cloning](https://arxiv.org/abs/1011.0686)
