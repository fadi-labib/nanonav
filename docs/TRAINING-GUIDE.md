# Training Guide

A practical guide to training the TRM navigation model, including recommended configurations, debugging tips, and optimization strategies.

## Quick Start Recipes

### Recipe 1: Fast Development Iteration

For quick experiments and debugging:

```bash
# Small dataset, small model, fast training
python scripts/generate_dataset.py --num-train 5000 --num-test 500

python -m trm_nav.train \
    --dim 64 \
    --depth 2 \
    --dropout 0.1 \
    --epochs 30 \
    --patience 10
```

**Expected time**: ~2-5 minutes on CPU
**Expected accuracy**: ~75-85%

### Recipe 2: Production Training (Recommended)

For best results:

```bash
# Large dataset with augmentation
python scripts/generate_dataset.py --num-train 50000 --num-test 2000 --augment

python -m trm_nav.train \
    --dim 128 \
    --depth 3 \
    --dropout 0.15 \
    --weight-decay 0.01 \
    --lr 5e-4 \
    --epochs 100 \
    --patience 20
```

**Expected time**: ~30-60 minutes on GPU, ~2-4 hours on CPU
**Expected accuracy**: ~90-95%

### Recipe 3: Maximum Performance

When accuracy is critical:

```bash
# Very large dataset
python scripts/generate_dataset.py --num-train 100000 --num-test 5000 --augment

python -m trm_nav.train \
    --dim 256 \
    --depth 4 \
    --dropout 0.2 \
    --weight-decay 0.02 \
    --lr 3e-4 \
    --epochs 150 \
    --patience 30 \
    --batch-size 128
```

**Expected time**: ~2-4 hours on GPU
**Expected accuracy**: ~95%+

---

## Parameter Tuning Guide

### Model Size Parameters

#### `--dim` (Hidden Dimension)

Controls the width of the model.

| Value | Use Case | Parameters |
|-------|----------|------------|
| 32 | Tiny model for debugging | ~15K |
| 64 | Default, good balance | ~50K |
| 128 | Better accuracy | ~200K |
| 256 | Maximum capacity | ~800K |

**Rule of thumb**: If underfitting, increase dim. If overfitting, decrease dim.

#### `--depth` (Number of Layers)

Controls the depth of the MLP-Mixer.

| Value | Use Case |
|-------|----------|
| 1 | Very simple patterns |
| 2 | Default, most tasks |
| 3 | Complex reasoning |
| 4+ | Diminishing returns for this task |

**Rule of thumb**: Start with 2. Only increase if val_acc plateaus without overfitting.

### Regularization Parameters

#### `--dropout` (Dropout Rate)

Probability of zeroing neurons during training.

| Value | When to Use |
|-------|-------------|
| 0.0 | When you have tons of data (100K+) |
| 0.1 | Default, mild regularization |
| 0.15-0.2 | When seeing overfitting |
| 0.3+ | Severe overfitting or very small data |

**Signs you need more dropout**:
- Train acc >> Val acc
- Val loss increasing while train loss decreasing

#### `--weight-decay` (L2 Regularization)

Penalty on large weight values.

| Value | When to Use |
|-------|-------------|
| 0.0 | When model is underfitting |
| 0.01 | Default |
| 0.05 | Moderate overfitting |
| 0.1 | Severe overfitting |

#### `--patience` (Early Stopping)

Number of epochs to wait for improvement.

| Value | When to Use |
|-------|-------------|
| 5-10 | Fast iteration, quick experiments |
| 15-20 | Default, production training |
| 30+ | Long training, large models |

### Learning Rate Parameters

#### `--lr` (Learning Rate)

How big of a step to take during optimization.

| Value | When to Use |
|-------|-------------|
| 1e-2 | Very small models, quick convergence |
| 1e-3 | Default |
| 5e-4 | Larger models (dim >= 128) |
| 1e-4 | Very large models, fine-tuning |

**Signs lr is too high**: Loss oscillates wildly, doesn't converge
**Signs lr is too low**: Training is very slow, loss decreases very gradually

#### `--batch-size` (Batch Size)

Number of samples per gradient update.

| Value | Trade-offs |
|-------|-----------|
| 32 | More noisy gradients, can escape local minima |
| 64 | Default, good balance |
| 128 | Smoother gradients, faster with GPU |
| 256+ | May need to increase lr proportionally |

**Rule**: If using GPU, larger batch sizes are more efficient.

---

## Training Diagnostics

### Understanding the Training Output

```
Epoch 1/100 - Train Loss: 1.5234, Train Acc: 0.3456 Val Loss: 1.4567, Val Acc: 0.3789
Epoch 2/100 - Train Loss: 1.2345, Train Acc: 0.5678 Val Loss: 1.1234, Val Acc: 0.5890
...
Epoch 50/100 - Train Loss: 0.0012, Train Acc: 0.9998 Val Loss: 2.3456, Val Acc: 0.7234 *
```

### Healthy Training Curves

```
Good: Both losses decrease, accuracies increase
┌──────────────────────────────────────┐
│ Loss                                 │
│ ╲                                    │
│  ╲___                                │
│      ╲____                           │
│           ╲____  ← Train Loss        │
│                ╲____                 │
│                     ╲____            │
│ ────────────────────────── Val Loss  │
└──────────────────────────────────────┘
```

### Overfitting Pattern

```
Bad: Val loss increases while train loss decreases
┌──────────────────────────────────────┐
│ Loss                                 │
│                          ___╱ Val    │
│                    _____╱            │
│               ____╱                  │
│ ╲        ____╱                       │
│  ╲______╱                            │
│         ╲____                        │
│              ╲____ Train             │
└──────────────────────────────────────┘
```

**Solutions**: More data, more augmentation, more dropout, more weight decay

### Underfitting Pattern

```
Bad: Both losses stay high, accuracies plateau early
┌──────────────────────────────────────┐
│ Loss                                 │
│ ────────────────────────── Val       │
│                                      │
│ ────────────────────────── Train     │
│                                      │
│ (Both stuck around 0.5-1.0)          │
└──────────────────────────────────────┘
```

**Solutions**: Bigger model (dim, depth), lower dropout, more training epochs, lower weight decay

---

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `--batch-size 32`
2. Reduce model size: `--dim 64 --depth 2`
3. Use CPU: `--device cpu`

### Issue: Training is very slow

**Solutions**:
1. Use GPU: `--device cuda`
2. Increase batch size: `--batch-size 128`
3. Reduce model size: `--dim 64`
4. Reduce max epochs: `--epochs 50`

### Issue: Loss is NaN

**Causes**:
- Learning rate too high
- Numerical instability

**Solutions**:
1. Lower learning rate: `--lr 1e-4`
2. Enable gradient clipping (already enabled by default)
3. Check for data issues

### Issue: Accuracy stuck at ~20%

This is random chance (1/5 actions).

**Solutions**:
1. Check data generation: Are paths valid?
2. Check encoding: Are tokens correct?
3. Try smaller model first to verify pipeline

### Issue: Perfect train accuracy but low validation

Classic overfitting.

**Solutions** (apply in order):
1. Enable augmentation: `--augment`
2. Increase dropout: `--dropout 0.2`
3. Increase weight decay: `--weight-decay 0.05`
4. Get more data: `--num-train 100000`
5. Reduce model size

---

## Data Quantity Guidelines

| Training Samples | Expected Val Accuracy | Notes |
|------------------|----------------------|-------|
| 1,000 | 50-65% | Severe overfitting |
| 5,000 | 65-80% | Default, may overfit |
| 10,000 | 75-85% | Good for development |
| 50,000 | 85-92% | Recommended |
| 100,000+ | 90-95%+ | Best results |

### With Augmentation

Augmentation effectively multiplies your data by 8×:

| Base Samples | Effective Samples | Expected Accuracy |
|--------------|------------------|-------------------|
| 5,000 | 40,000 | ~85% |
| 10,000 | 80,000 | ~90% |
| 50,000 | 400,000 | ~95% |

---

## Experiment Tracking

### Recommended Workflow

```bash
# Create experiment directory
mkdir -p experiments/exp001

# Generate data
python scripts/generate_dataset.py --num-train 50000 --augment \
    --output-dir experiments/exp001/data

# Train with specific checkpoint dir
python -m trm_nav.train \
    --train-path experiments/exp001/data/train.pt \
    --val-path experiments/exp001/data/test.pt \
    --checkpoint-dir experiments/exp001/checkpoints \
    --dim 128 --depth 3 --dropout 0.15

# Results will be in:
# - experiments/exp001/checkpoints/best.pt
# - experiments/exp001/checkpoints/history.json
```

### Comparing Experiments

```bash
# Quick comparison
for exp in experiments/exp*/; do
    echo "=== $exp ==="
    python -c "
import json
with open('${exp}checkpoints/history.json') as f:
    h = json.load(f)
    print(f'Best val acc: {max(h[\"val_acc\"]):.4f}')
    print(f'Final val acc: {h[\"val_acc\"][-1]:.4f}')
"
done
```

---

## GPU vs CPU Training

### CPU Training

```bash
python -m trm_nav.train --device cpu
```

**Pros**: Always available, no CUDA setup
**Cons**: 5-10x slower

### GPU Training

```bash
# Auto-detect
python -m trm_nav.train

# Force CUDA
python -m trm_nav.train --device cuda
```

**Verify GPU is being used**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

### Typical Training Times (50K samples)

| Config | GPU (RTX 3080) | CPU (8 cores) |
|--------|----------------|---------------|
| dim=64, depth=2 | ~10 min | ~60 min |
| dim=128, depth=3 | ~30 min | ~3 hours |
| dim=256, depth=4 | ~90 min | ~8 hours |

---

## Checkpointing

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 45,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': 0.92,
    'val_loss': 0.23,
    'config': {
        'grid_size': 8,
        'dim': 128,
        'depth': 3,
        'dropout': 0.15
    }
}
```

### Loading a Checkpoint

```python
checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')

# Recreate model with same config
model = create_model(
    grid_size=checkpoint['config']['grid_size'],
    dim=checkpoint['config']['dim'],
    depth=checkpoint['config']['depth'],
    dropout=checkpoint['config']['dropout']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
```

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pt')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer (important for momentum)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue from epoch 51
start_epoch = checkpoint['epoch'] + 1
```

---

## Hyperparameter Search

### Grid Search Example

```bash
for dim in 64 128 256; do
    for depth in 2 3 4; do
        for dropout in 0.1 0.15 0.2; do
            echo "Testing dim=$dim depth=$depth dropout=$dropout"
            python -m trm_nav.train \
                --dim $dim \
                --depth $depth \
                --dropout $dropout \
                --checkpoint-dir checkpoints/grid_${dim}_${depth}_${dropout} \
                --epochs 50
        done
    done
done
```

### Recommended Search Ranges

| Parameter | Search Range | Step |
|-----------|-------------|------|
| dim | [64, 128, 256] | 2x |
| depth | [2, 3, 4] | 1 |
| dropout | [0.1, 0.15, 0.2, 0.25] | 0.05 |
| lr | [1e-3, 5e-4, 1e-4] | /2 |
| weight_decay | [0.01, 0.02, 0.05] | 2x |

---

## Reproducibility

### Setting Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Reproducibility Checklist

- [x] Set random seeds for all libraries
- [x] Use deterministic data loading (`num_workers=0`)
- [x] Document exact command used
- [x] Save model config in checkpoint
- [x] Record library versions

---

## Final Recommendations

1. **Start small**: Begin with dim=64, depth=2, 5K samples
2. **Verify pipeline**: Ensure A* paths are correct, encoding works
3. **Scale up gradually**: Increase data first, then model size
4. **Always use augmentation**: Free 8× data improvement
5. **Monitor for overfitting**: Watch val_loss, not just val_acc
6. **Save experiments**: Keep checkpoints and configs organized
