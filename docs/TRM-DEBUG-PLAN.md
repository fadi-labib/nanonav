# TRM Debug Plan: Fixing Plateau at 0.25

## Problem Statement

- **Symptom**: TRM model plateaus at 0.25 accuracy (random chance for 4-class classification)
- **Working**: MLP fallback trains and converges correctly
- **Conclusion**: Bug is specific to TRM algorithm, not data pipeline or training loop fundamentals

## Reference Implementation

Official repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

---

## Debug Tools Available

### 1. Training with Debug Mode
```bash
# Run training with full debug output
python -m trm_nav.train --debug --epochs 10 --no-resume

# Debug output includes:
# - Initial model analysis
# - Per-epoch gradient norms and summaries
# - Prediction entropy tracking
# - Class distribution monitoring
# - Automatic plots every 5 epochs
```

### 2. Standalone Debug Script
```bash
# Analyze a trained checkpoint
python scripts/debug_trm.py --checkpoint checkpoints/best.pt --full

# Analyze fresh model (no checkpoint)
python scripts/debug_trm.py --no-checkpoint --full

# Compare TRM vs Fallback
python scripts/debug_trm.py --no-checkpoint --use-fallback
```

### 3. Debug Output Location
All debug output is saved to `debug_output/`:
- `debug_epoch_N.png` - Plots for each epoch
- `final_debug_plots.png` - Final comprehensive plots
- `final_debug_stats.json` - Raw statistics in JSON format

---

## Phase 1: Gradient Flow Analysis

### 1.1 Verify Gradients Flow Through Recursion
```python
# Add this diagnostic to training loop
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
        else:
            print(f"{name}: NO GRADIENT")
```

### 1.2 Check for Vanishing/Exploding Gradients
- Monitor gradient norms per layer during training
- Look for NaN or zero gradients after recursion steps
- Key question: Does gradient magnitude decrease with more recursion steps?

### 1.3 Specific Checks
- [ ] Embedding layer receives gradients
- [ ] MLP-Mixer layers receive gradients
- [ ] Recursive loop doesn't use `.detach()` anywhere
- [ ] No `torch.no_grad()` context around recursion

---

## Phase 2: Compare with Official Implementation

### 2.1 Files to Study in `external/trm/`
| File | What to Look For |
|------|------------------|
| `dataset/build_maze_dataset.py` | Input encoding scheme |
| `models/recursive_reasoning/trm.py` | Core TRM architecture |
| `models/layers.py` | MLP-Mixer implementation |
| `models/losses.py` | Loss function (per-step vs final) |
| `pretrain.py` | Training loop and optimizer settings |

### 2.2 Key Differences to Identify
- [ ] Input tokenization/encoding
- [ ] How recursive state is initialized
- [ ] How recursive state is updated (residual connections?)
- [ ] Loss computation: per-iteration supervision vs final-only
- [ ] Optimizer: they use `adam-atan2`, we use AdamW
- [ ] Learning rate and scheduler

### 2.3 Commands to Compare
```bash
# Diff our TRM vs official TRM
diff -u external/trm/models/recursive_reasoning/trm.py trm_nav/official_trm/trm.py

# Check their training script
cat external/trm/pretrain.py | head -200
```

---

## Phase 3: Architecture Audit

### 3.1 Check Navigation TRM Wrapper
**File**: `trm_nav/official_trm/navigation_trm_submodule.py`

Questions:
- [ ] Is output head receiving proper representations?
- [ ] Is mean pooling appropriate? (vs CLS token, vs last token)
- [ ] Does recursive refinement actually modify hidden state?
- [ ] Are layer norms in correct positions?

### 3.2 Check TRM Core
**File**: `trm_nav/official_trm/trm.py`

Compare with: `external/trm/models/recursive_reasoning/trm.py`

- [ ] Recursive loop structure matches
- [ ] Residual connections present
- [ ] Correct tensor shapes throughout
- [ ] No accidental tensor copies breaking gradients

### 3.3 Model Wrapper
**File**: `trm_nav/model.py`

- [ ] TRM instantiated correctly
- [ ] Forward pass returns correct tensor
- [ ] Device handling doesn't break computation graph

---

## Phase 4: Training Loop Audit

### 4.1 Check Training Script
**File**: `trm_nav/train.py`

- [ ] Loss function matches official (CrossEntropy?)
- [ ] Check for auxiliary losses (per-recursion supervision)
- [ ] Optimizer configuration
- [ ] Gradient clipping settings
- [ ] Learning rate schedule

### 4.2 Official Training Settings (from paper/repo)
```python
# Likely settings from official repo - verify these
optimizer = AdamAtan2(lr=1e-3)  # Not standard AdamW
# May use per-step loss supervision
# May use different batch sizes
```

---

## Phase 5: Quick Diagnostic Tests

### 5.1 Overfit Test (Critical)
```bash
# If TRM can't overfit on 10 samples, architecture is fundamentally broken
python -m trm_nav.train --num-train 10 --epochs 1000 --batch-size 10
```
- Expected: Should reach ~100% train accuracy
- If fails: Architecture bug confirmed

### 5.2 Single Recursion Test
```bash
# Test with recursion=1 (should behave like regular network)
python -m trm_nav.train --max-recursion 1
```
- If this works: Bug is in recursion loop
- If this fails: Bug is elsewhere

### 5.3 Hidden State Inspection
```python
# Add to forward pass temporarily
def forward(self, x):
    h = self.embed(x)
    for i in range(self.num_recursions):
        h_before = h.clone()
        h = self.mixer(h)
        diff = (h - h_before).abs().mean()
        print(f"Recursion {i}: state_change={diff.item():.6f}")
    return self.head(h)
```
- If diff ≈ 0: Recursion is a no-op
- If diff explodes: Instability issue

### 5.4 Direct Comparison
```python
# Compare TRM vs MLP on identical input
x = torch.randint(0, 10, (1, 68))
trm_out = trm_model(x)
mlp_out = mlp_model(x)
print(f"TRM output stats: mean={trm_out.mean()}, std={trm_out.std()}")
print(f"MLP output stats: mean={mlp_out.mean()}, std={mlp_out.std()}")
```

---

## Most Likely Causes (Ranked by Probability)

| Rank | Cause | How to Verify |
|------|-------|---------------|
| 1 | Gradient not flowing through recursion | Phase 1 gradient checks |
| 2 | Output head receives garbage/zeros | Phase 5.3 hidden state check |
| 3 | Hidden state not updating in loop | Phase 5.3 state diff check |
| 4 | Missing per-step loss supervision | Phase 2 loss comparison |
| 5 | Wrong tokenization vs official | Phase 2 encoding comparison |
| 6 | Optimizer mismatch (AdamW vs Atan2) | Phase 4 optimizer check |

---

## Action Checklist

### Quick Wins (Try First)
- [ ] Run overfit test on 10 samples
- [ ] Run single recursion test (--max-recursion 1)
- [ ] Add gradient norm logging
- [ ] Print hidden state changes per recursion

### Deep Investigation (If Quick Wins Don't Help)
- [ ] Full diff against official TRM implementation
- [ ] Study official maze training script
- [ ] Check if official uses per-step losses
- [ ] Try official optimizer (adam-atan2)

### Nuclear Options (Last Resort)
- [ ] Copy official TRM code directly, adapt minimally
- [ ] Train official TRM on their maze task to verify it works
- [ ] Step-by-step debug with breakpoints

---

## Notes

- Date created: 2025-12-05
- MLP fallback confirms: data pipeline, training loop, and evaluation are correct
- The bug is isolated to TRM-specific code paths
