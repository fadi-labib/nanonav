# NanoNav: Tiny Recursive Model for Local Navigation

A proof-of-concept implementation comparing Tiny Recursive Models (TRM) to A* for grid-based navigation. This project demonstrates how to train a neural network to imitate optimal pathfinding behavior using behavioral cloning.

## Goal

Prove that TRM can achieve **comparable results to A*** on grid navigation tasks:
- Success rate ≥ 85%
- Path length ratio ≤ 1.3 (TRM path / A* optimal path)

## Background & Motivation

### The Tiny Recursive Model (TRM)

This project is based on the **Tiny Recursive Model (TRM)** architecture introduced by Samsung SAIL Montreal:

> **"Less is More: Recursive Reasoning with Tiny Networks"**
> Alexia Jolicoeur-Martineau, Samsung SAIL Montreal, 2025
> [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
>
> Official implementation: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

The key insight of TRM is that **recursive computation can substitute for model depth**. Instead of building deeper networks with more parameters, TRM applies a smaller network multiple times to iteratively refine its representations. This mimics how humans often "think twice" about a problem, revisiting and refining their reasoning.

### Core Concept: Recursive Refinement

```
Traditional Deep Network:          TRM Approach:

Input → Layer1 → Layer2 → ...      Input → Small Network ─┐
      → LayerN → Output                    ↑              │
                                           └──── Loop N times
(Many parameters, single pass)            ↓
                                         Output
                                   (Fewer parameters, multiple passes)
```

**How it works:**
1. Input is embedded into a latent representation
2. A small network (MLP-Mixer) processes the representation
3. The output is fed back as input for another refinement pass
4. After N iterations, the refined representation is used for prediction

### Why TRM for Navigation?

Navigation is an interesting testbed for recursive reasoning because:

1. **Iterative Decision Making**: Real navigation involves continuously reassessing your position relative to the goal. TRM's recursive passes mirror this "look-think-adjust" loop.

2. **Spatial Reasoning**: Finding a path requires understanding spatial relationships between obstacles, current position, and goal. Multiple refinement passes allow the model to propagate information across the grid representation.

3. **Efficiency vs. Capability Trade-off**: Traditional pathfinding (A*) is optimal but requires explicit graph search. Can a tiny learned model with recursive refinement achieve comparable results? This is the core research question.

4. **Resource-Constrained Deployment**: For embedded robotics (the "nano" in NanoNav), having a small model that can "think longer" on hard problems is more practical than a large model that's fast but memory-hungry.

### Research Questions

This PoC explores:

- Can behavioral cloning from A* demonstrations teach spatial reasoning?
- Does recursive refinement help with navigation decisions?
- What's the trade-off between model size, recursion depth, and accuracy?
- How does a learned policy generalize to unseen grid configurations?

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate training data (with augmentation for better results)
python scripts/generate_dataset.py --num-train 50000 --augment

# 4. Train the model
python -m trm_nav.train --dim 128 --depth 3 --dropout 0.15

# 5. Visual test
python scripts/test_trm_visual.py --checkpoint checkpoints/best.pt

# 6. Full benchmark
python scripts/run_benchmark.py --checkpoint checkpoints/best.pt
```

## Project Structure

```
trm_nav/
├── trm_nav/                   # Main package
│   ├── __init__.py
│   ├── a_star.py              # A* pathfinding algorithm (teacher/oracle)
│   ├── map_generator.py       # Random solvable map generation
│   ├── dataset.py             # Dataset creation with augmentation
│   ├── model.py               # TRM model wrapper (requires tiny-recursive-model)
│   ├── train.py               # Training loop with regularization
│   ├── evaluate.py            # Benchmarking and rollouts
│   └── visualize.py           # ASCII and matplotlib visualization
├── scripts/
│   ├── generate_dataset.py    # Dataset generation script
│   ├── run_benchmark.py       # Full evaluation benchmark
│   ├── test_astar_visual.py   # Visual A* test
│   └── test_trm_visual.py     # Visual TRM vs A* comparison
├── tests/                     # Unit tests
├── docs/                      # Documentation
│   ├── IMPLEMENTATION-NOTES.md
│   └── TRAINING-GUIDE.md
├── data/                      # Generated datasets (*.pt files)
├── checkpoints/               # Saved model checkpoints
└── results/                   # Benchmark outputs
```

## How It Works

### 1. Data Generation (Behavioral Cloning)

```
Random Grid → A* Optimal Path → (state, action) pairs
```

- Generate random 8×8 grids with ~20% obstacle density
- Ensure start and goal are reachable (solvable maps only)
- Run A* to get the optimal path
- For each position along the path, create a training sample:
  - **Input**: Current grid state + current position + goal position
  - **Output**: Optimal action to take (from A*)

### 2. State Encoding

```
Grid (8×8) → Flatten → 64 tokens
             + 4 coordinate tokens (start_row, start_col, goal_row, goal_col)
             = 68 tokens total

Token values:
  - 1 = Free cell
  - 2 = Obstacle
  - 3+ = Coordinate values (offset by 3 to avoid collision)
```

### 3. Model Architecture

```
Input: 68 tokens
       ↓
Embedding: tokens → (batch, 68, dim)
       ↓
MLP-Mixer: depth × [TokenMixing + ChannelMixing]
       ↓
Recursive Refinement: N iterations of the mixer
       ↓
Mean Pooling: (batch, 68, dim) → (batch, dim)
       ↓
Classifier: LayerNorm → Dropout → Linear → GELU → Dropout → Linear
       ↓
Output: 5 action logits (UP, DOWN, LEFT, RIGHT, STAY)
```

### 4. Training

- **Loss**: Cross-entropy between predicted and optimal actions
- **Optimizer**: AdamW with weight decay (L2 regularization)
- **Scheduler**: Cosine annealing learning rate
- **Regularization**: Dropout + early stopping

### 5. Inference (Rollout)

```python
while current_position != goal:
    tokens = encode_state(grid, current_position, goal)
    action = model.predict_action(tokens)
    current_position = execute_action(current_position, action)
```

## Key Features

### Data Augmentation (8× more data)

Enable with `--augment` flag. Generates 8 versions of each sample:

| Transformation | Grid Operation | Action Mapping |
|----------------|----------------|----------------|
| Original | - | - |
| Rotate 90° CW | `np.rot90(grid, k=-1)` | UP→RIGHT, RIGHT→DOWN, DOWN→LEFT, LEFT→UP |
| Rotate 180° | `np.rot90(grid, k=2)` | UP→DOWN, DOWN→UP, LEFT→RIGHT, RIGHT→LEFT |
| Rotate 270° CW | `np.rot90(grid, k=-3)` | UP→LEFT, LEFT→DOWN, DOWN→RIGHT, RIGHT→UP |
| Flip Horizontal | `np.fliplr(grid)` | LEFT↔RIGHT |
| Flip Vertical | `np.flipud(grid)` | UP↔DOWN |
| Diagonal 1 | flip_h + rot90 | Combined |
| Diagonal 2 | flip_v + rot90 | Combined |

**Why augmentation helps**: The model learns rotational and reflective invariance, meaning it understands that navigation principles are the same regardless of orientation.

### Regularization (Prevents Overfitting)

| Technique | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| Dropout | `--dropout` | 0.1 | Randomly zeros neurons during training |
| Weight Decay | `--weight-decay` | 0.01 | L2 penalty on weights (AdamW) |
| Early Stopping | `--patience` | 15 | Stops when val_loss stops improving |
| Gradient Clipping | - | 1.0 | Prevents exploding gradients |

### Model Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Dimension | `--dim` | 64 | Hidden layer size (64, 128, 256) |
| Depth | `--depth` | 2 | Number of MLP-Mixer layers |
| Recursion | `max_recursion_steps` | 8 | TRM refinement iterations |

**Scaling guidance**:
- Small (fast): `--dim 64 --depth 2` (~50K params)
- Medium: `--dim 128 --depth 3` (~200K params)
- Large: `--dim 256 --depth 4` (~800K params)

## Command Reference

### Dataset Generation

```bash
# Basic (5000 train, 500 test)
python scripts/generate_dataset.py

# With augmentation (recommended)
python scripts/generate_dataset.py --num-train 50000 --augment

# All options
python scripts/generate_dataset.py \
    --num-train 50000 \
    --num-test 1000 \
    --grid-size 8 \
    --obstacle-density 0.2 \
    --train-seed 42 \
    --test-seed 99999 \
    --output-dir data \
    --augment
```

### Training

```bash
# Basic training
python -m trm_nav.train

# Recommended settings (after overfitting fix)
python -m trm_nav.train \
    --dim 128 \
    --depth 3 \
    --dropout 0.15 \
    --weight-decay 0.01 \
    --lr 5e-4 \
    --patience 15 \
    --epochs 100

# All options
python -m trm_nav.train \
    --train-path data/train.pt \
    --val-path data/test.pt \
    --checkpoint-dir checkpoints \
    --grid-size 8 \
    --dim 128 \
    --depth 3 \
    --dropout 0.15 \
    --batch-size 64 \
    --lr 5e-4 \
    --weight-decay 0.01 \
    --epochs 100 \
    --patience 15 \
    --device cuda
```

### Testing & Visualization

```bash
# Visual A* test (verify pathfinding works)
python scripts/test_astar_visual.py --seed 42 --size 8

# Visual TRM test (compare TRM to A*)
python scripts/test_trm_visual.py --checkpoint checkpoints/best.pt --seed 42

# Save comparison image
python scripts/test_trm_visual.py --checkpoint checkpoints/best.pt --save output.png

# ASCII only (no matplotlib)
python scripts/test_trm_visual.py --checkpoint checkpoints/best.pt --no-plot
```

### Benchmarking

```bash
# Full benchmark (100 episodes)
python scripts/run_benchmark.py --checkpoint checkpoints/best.pt

# Custom benchmark
python scripts/run_benchmark.py \
    --checkpoint checkpoints/best.pt \
    --episodes 500 \
    --grid-size 8 \
    --seed 12345
```

## Understanding Training Output

```
Epoch 45/100 - Train Loss: 0.1234, Train Acc: 0.9567 Val Loss: 0.2345, Val Acc: 0.9123 *
```

| Metric | Description | Good Values |
|--------|-------------|-------------|
| Train Loss | Cross-entropy on training set | ↓ Lower is better |
| Train Acc | % correct actions on training set | ↑ Higher is better |
| Val Loss | Cross-entropy on held-out test set | ↓ Lower is better |
| Val Acc | % correct actions on test set | ↑ Higher is better |
| `*` | Indicates new best model saved | - |

### Diagnosing Problems

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Train Acc 100%, Val Acc ~70% | Overfitting | ↑ dropout, ↑ weight_decay, ↑ data |
| Train/Val Acc both low | Underfitting | ↑ dim, ↑ depth, ↓ dropout |
| Val Loss increasing | Overfitting | Early stopping will trigger |
| Training very slow | - | ↓ dim, use GPU (`--device cuda`) |

## Troubleshooting

### TRM Library API Issue

If you see:
```
TypeError: TinyRecursiveModel.forward() missing 2 required positional arguments
```

**Cause**: The `tiny-recursive-model` library is designed for sequence-to-sequence tasks, not classification.

**Solution**: The model wrapper (`model.py`) uses the internal components directly:
```python
# Instead of: output = self.trm(tokens)
# We use:
embedded = self.trm.input_embed(tokens)
features = embedded
for _ in range(self.max_recursion_steps):
    features = self.trm.network(features)
features = features.mean(dim=1)  # Pool for classification
```

### Overfitting (Train Acc >> Val Acc)

**Symptoms**:
```
Epoch 50 - Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 4.2700, Val Acc: 0.6900
```

**Solutions**:
1. Enable data augmentation: `--augment`
2. Increase dropout: `--dropout 0.2`
3. Increase weight decay: `--weight-decay 0.05`
4. Use more training data: `--num-train 100000`
5. Use smaller model: `--dim 64 --depth 2`

### No GPU Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python -m trm_nav.train --device cpu
```

## Dependencies

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
networkx>=2.6.0
tqdm>=4.62.0
tiny-recursive-model>=0.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

## File Descriptions

### Core Modules

| File | Purpose |
|------|---------|
| `a_star.py` | A* pathfinding implementation using NetworkX. Provides optimal paths as training targets. |
| `map_generator.py` | Generates random grids with guaranteed solvability (start and goal always reachable). |
| `dataset.py` | Converts A* demonstrations to (state, action) pairs. Includes data augmentation (rotations, flips). |
| `model.py` | TRM model wrapper. Uses MLP-Mixer with recursive refinement. Falls back to simple MLP if TRM not installed. |
| `train.py` | Training loop with AdamW optimizer, cosine LR schedule, early stopping, and gradient clipping. |
| `evaluate.py` | Runs complete navigation episodes and computes success rate / path ratio metrics. |
| `visualize.py` | ASCII grid printing and matplotlib path visualization. |

### Scripts

| Script | Purpose |
|--------|---------|
| `generate_dataset.py` | CLI for dataset generation with configurable size, density, augmentation. |
| `run_benchmark.py` | Runs full evaluation and prints success metrics. |
| `test_astar_visual.py` | Visual test of A* pathfinding (sanity check). |
| `test_trm_visual.py` | Side-by-side TRM vs A* comparison with visualization. |

## Concepts Explained

### Behavioral Cloning
Training a neural network to imitate an expert (A*) by learning from (state, action) demonstrations. The network learns the policy π(state) → action without understanding *why* those actions are optimal.

### MLP-Mixer
An architecture that mixes information across both the token dimension (which tokens interact) and the channel dimension (which features interact). Simpler than Transformers but effective for fixed-size inputs.

### Recursive Refinement
TRM applies the same network multiple times to iteratively refine representations. This can help with problems requiring multi-step reasoning, though for simple navigation the benefit may be limited.

### Early Stopping
Monitoring validation loss and stopping training when it stops improving. Prevents the model from memorizing training data (overfitting) and helps it generalize to new grids.

## Expected Results

After proper training with augmentation:

```
=== NanoNav Benchmark Results ===

Grid Size: 8x8
Episodes:  100

Agent       Success    Avg Ratio    Timeouts
--------------------------------------------
A*          100.0%     1.00         0
TRM         ~90%+      ~1.1-1.2     ~5-10

Success Criteria:
  Success Rate >= 85%: PASS ✓
  Path Ratio <= 1.3:   PASS ✓
```

## Next Steps

If PoC succeeds:
1. Scale to 16×16 and 32×32 grids
2. Add baseline comparisons (MLP, CNN, Transformer)
3. Ablation study on recursion depth
4. Test with dynamic obstacles
5. Transfer to continuous action spaces

## Acknowledgments

Special thanks to my friend [Claude](https://claude.ai) by Anthropic, who helped me code and learn faster throughout this project.

## License

MIT
