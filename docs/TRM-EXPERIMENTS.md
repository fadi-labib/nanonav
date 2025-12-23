# TRM Navigation Experiments

This document summarizes the experiments conducted to train the TRM (Tiny Recursive Model) for path prediction on 8x8 navigation grids.

## Problem Statement

Train TRM to predict the full path from start to goal on an 8x8 grid with obstacles. This is a sequence-to-sequence task where each cell is classified as: pad (0), free (1), obstacle (2), or path (3).

## Key Findings

### 1. Gradient Flow Issues

**Problem:** Initial training plateaued at 25% accuracy (random for 4 classes).

**Root Cause:** Vanishing gradients with high recursion steps.

**Solutions:**
- Reduced `max_recursion` from 8 to 2 - improved accuracy to 37%
- Implemented `--grad-last-only` flag (like original TRM) - allows higher recursion without vanishing gradients

| max_recursion | grad_last_only | Accuracy |
|---------------|----------------|----------|
| 8 | No | 25% (plateau) |
| 2 | No | 37% |
| 8 | Yes | 49% |

### 2. Class Imbalance

**Problem:** Model ignoring minority path class (0% path recall).

**Solutions tried:**

| Loss Function | Class Weight | Accuracy | Path Recall |
|---------------|--------------|----------|-------------|
| CrossEntropy | 1x | 89% | 0% |
| CrossEntropy | 5x | 78-85% | 22-49% |
| CrossEntropy | 8x | 49% | 87% |
| CrossEntropy | 15x | 32-35% | 97-99% |
| Focal (gamma=2) | 8x | 37.9% | 98.7% |

**Best balance:** CrossEntropy with 8x class weight for path class.

### 3. L_cycles Parameter

**Discovery:** Original TRM uses `L_cycles=4-6` for inner refinement loop. We had it hardcoded to 1.

| H_cycles | L_cycles | Accuracy | Path Recall | Notes |
|----------|----------|----------|-------------|-------|
| 8 | 1 | 49.2% | 87.4% | Previous best |
| 3 | 4 | 49.7% | 86.3% | Matches original TRM config |

### 4. Training Stability

**Issue:** Training collapsed at epoch 50 with `weight_decay=1.0`, jumping to 100% path recall / 35.5% accuracy.

**Recommendation:** Use `weight_decay=0.1` instead of 1.0.

## Best Configuration

```bash
python -m trm_nav.train_path \
  --epochs 500 \
  --max-recursion 3 \
  --l-cycles 4 \
  --batch-size 16384 \
  --num-workers 8 \
  --lr 0.001 \
  --weight-decay 0.1 \
  --grad-last-only \
  --patience 30
```

## GPU Optimizations

Implemented for A6000 (48GB) / RTX 4080 (16GB):

| Optimization | Impact |
|--------------|--------|
| Mixed precision (AMP) | ~2x faster |
| torch.compile | ~50% faster after warmup |
| TensorFloat32 matmul | Faster on Ampere+ |
| Batch size 16384 | Better GPU utilization |
| 8 data workers | Reduced CPU bottleneck |
| Pin memory + persistent workers | Faster data transfer |

**Throughput:** ~35,000 samples/sec on RTX 4080

## Architecture Differences from Original TRM

| Feature | Original TRM | Our Implementation |
|---------|--------------|-------------------|
| H_cycles | 3 | Configurable (default 8) |
| L_cycles | 4-6 | Configurable (default 1) |
| Gradient flow | Last H-cycle only | All or last only (flag) |
| Weight decay | 1.0 | 0.01-0.1 |
| EMA | Yes | Not implemented |
| Epochs | 50,000 | 100-500 |

## Metrics Explained

- **Accuracy:** % of all cells correctly classified (pad, free, obstacle, path)
- **Path Recall:** % of actual path cells correctly predicted as path
- **Loss:** CrossEntropy or Focal loss value

## Files Modified

- `trm_nav/official_trm/navigation_trm_submodule.py` - TRM wrapper with grad_last_only, l_cycles
- `trm_nav/model.py` - Model factory with new parameters
- `trm_nav/train_path.py` - Training script with all optimizations
- `trm_nav/dataset.py` - Path prediction dataset (4-class encoding)

## Next Steps

1. Implement EMA (Exponential Moving Average) like original TRM
2. Train for more epochs (original uses 50,000)
3. Try different H_cycles/L_cycles combinations
4. Investigate training instability with high weight_decay
