#!/usr/bin/env python3
"""
TRM Debug Script

Quick diagnosis tool for investigating TRM model internals.
Use this to debug why TRM might be plateauing at random chance.

Usage:
    # Diagnose a trained model
    python scripts/debug_trm.py --checkpoint checkpoints/best.pt

    # Diagnose without a checkpoint (fresh model)
    python scripts/debug_trm.py --no-checkpoint

    # Full analysis with gradient check
    python scripts/debug_trm.py --checkpoint checkpoints/best.pt --full
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path

from trm_nav.model import create_model
from trm_nav.dataset import NavigationDataset


def analyze_model_structure(model):
    """Analyze model structure and parameter counts."""
    print("\n" + "="*70)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*70)

    total_params = 0
    trainable_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        layer_info.append((name, param.shape, param.numel(), param.requires_grad))

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable: {total_params - trainable_params:,}")

    print("\nLayer breakdown:")
    for name, shape, count, trainable in layer_info:
        status = "✓" if trainable else "✗"
        print(f"  {status} {name}: {list(shape)} ({count:,})")


def analyze_forward_pass(model, tokens, device):
    """Analyze a single forward pass."""
    print("\n" + "="*70)
    print("FORWARD PASS ANALYSIS")
    print("="*70)

    model.eval()
    tokens = tokens.to(device)

    # Check if model has debug forward
    inner_model = model.trm if hasattr(model, 'trm') else model
    if hasattr(inner_model, 'forward_with_debug'):
        print("\nRunning debug forward pass...")
        with torch.no_grad():
            logits, debug_info = inner_model.forward_with_debug(tokens)

        inner_model.print_debug_summary(debug_info)

        # Additional analysis
        print("\n[ADDITIONAL ANALYSIS]")

        # Check for recursion effectiveness
        if debug_info.step_changes:
            changes = debug_info.step_changes
            print(f"\n  Recursion effectiveness:")
            print(f"    First 5 step changes: {[f'{c:.4f}' for c in changes[:5]]}")
            print(f"    Last 5 step changes:  {[f'{c:.4f}' for c in changes[-5:]]}")

            if changes[-1] < changes[0] * 0.01:
                print("    → Recursion converged (later steps have minimal effect)")
            elif changes[-1] > changes[0]:
                print("    → Recursion DIVERGING (potential instability)")
            else:
                print("    → Recursion active throughout")

        return logits
    else:
        print("\nRunning standard forward pass...")
        with torch.no_grad():
            logits = model(tokens)

        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits stats:")
        print(f"  Mean: {logits.mean().item():.4f}")
        print(f"  Std:  {logits.std().item():.4f}")
        print(f"  Min:  {logits.min().item():.4f}")
        print(f"  Max:  {logits.max().item():.4f}")
        print(f"  Range: {(logits.max() - logits.min()).item():.4f}")

        probs = torch.softmax(logits, dim=-1)
        print(f"\nPrediction probabilities (sample 0): {probs[0].tolist()}")

        return logits


def analyze_gradient_flow(model, tokens, targets, device):
    """Analyze gradient flow through the model."""
    print("\n" + "="*70)
    print("GRADIENT FLOW ANALYSIS")
    print("="*70)

    model.train()
    tokens = tokens.to(device)
    targets = targets.to(device)

    # Zero gradients
    model.zero_grad()

    # Forward pass
    logits = model(tokens)
    loss = nn.CrossEntropyLoss()(logits, targets)

    print(f"\nLoss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Analyze gradients
    grad_info = []
    zero_grad_layers = []
    nan_grad_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            grad_info.append((name, grad_norm, grad_mean, has_nan, has_inf))

            if grad_norm < 1e-10:
                zero_grad_layers.append(name)
            if has_nan:
                nan_grad_layers.append(name)
        else:
            grad_info.append((name, None, None, False, False))

    # Sort by gradient norm
    grad_info_sorted = sorted(
        [(n, norm, m, nan, inf) for n, norm, m, nan, inf in grad_info if norm is not None],
        key=lambda x: x[1],
        reverse=True
    )

    print("\nGradient norms (highest first):")
    for name, norm, mean, has_nan, has_inf in grad_info_sorted[:10]:
        flags = ""
        if has_nan:
            flags += " [NaN!]"
        if has_inf:
            flags += " [Inf!]"
        short_name = '.'.join(name.split('.')[-3:])
        print(f"  {norm:12.2e}  {short_name}{flags}")

    print("\nGradient norms (lowest first):")
    for name, norm, mean, has_nan, has_inf in grad_info_sorted[-10:]:
        flags = ""
        if norm < 1e-10:
            flags += " [ZERO!]"
        short_name = '.'.join(name.split('.')[-3:])
        print(f"  {norm:12.2e}  {short_name}{flags}")

    # Warnings
    if zero_grad_layers:
        print(f"\n⚠️  WARNING: {len(zero_grad_layers)} layers have zero gradients!")
        print("   This suggests gradients are not flowing through these layers.")
        print("   Affected layers:")
        for name in zero_grad_layers[:5]:
            print(f"     - {name}")

    if nan_grad_layers:
        print(f"\n🚨 CRITICAL: {len(nan_grad_layers)} layers have NaN gradients!")
        print("   This indicates numerical instability.")

    # Check embedding gradients specifically
    print("\nKey layer gradient check:")
    for name, norm, mean, has_nan, has_inf in grad_info:
        if norm is not None and ('embed' in name.lower() or 'classifier' in name.lower()):
            status = "✓" if norm > 1e-8 else "✗"
            print(f"  {status} {name}: {norm:.2e}")


def analyze_predictions(model, dataloader, device, num_batches=10):
    """Analyze prediction distribution."""
    print("\n" + "="*70)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*70)

    model.eval()
    all_preds = []
    all_targets = []
    all_correct = []

    with torch.no_grad():
        for i, (tokens, targets) in enumerate(dataloader):
            if i >= num_batches:
                break

            tokens = tokens.to(device)
            targets = targets.to(device)

            logits = model(tokens)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_correct.extend((preds == targets).cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_correct = np.array(all_correct)

    print(f"\nSamples analyzed: {len(all_preds)}")
    print(f"Overall accuracy: {all_correct.mean():.4f}")

    # Class distribution
    num_classes = 4
    print(f"\nPredicted class distribution:")
    for c in range(num_classes):
        count = (all_preds == c).sum()
        pct = count / len(all_preds) * 100
        bar = "█" * int(pct / 5)
        print(f"  Class {c}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nTarget class distribution:")
    for c in range(num_classes):
        count = (all_targets == c).sum()
        pct = count / len(all_targets) * 100
        bar = "█" * int(pct / 5)
        print(f"  Class {c}: {count:5d} ({pct:5.1f}%) {bar}")

    # Check for collapse
    pred_dist = np.array([(all_preds == c).sum() for c in range(num_classes)]) / len(all_preds)
    if max(pred_dist) > 0.8:
        dominant = np.argmax(pred_dist)
        print(f"\n⚠️  WARNING: Model collapsed to predicting class {dominant} ({pred_dist[dominant]*100:.1f}%)")
    elif max(pred_dist) - min(pred_dist) < 0.1:
        print(f"\n⚠️  WARNING: Model predicting near-uniform distribution (random guessing)")

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            acc = all_correct[mask].mean()
            print(f"  Class {c}: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Debug TRM model internals")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true", help="Analyze fresh model without checkpoint")
    parser.add_argument("--data-path", type=str, default="data/train.pt", help="Path to training data")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max-recursion", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--full", action="store_true", help="Run full analysis including gradient flow")
    parser.add_argument("--use-fallback", action="store_true", help="Test fallback MLP model")

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        grid_size=args.grid_size,
        dim=args.dim,
        max_recursion_steps=args.max_recursion,
        use_fallback=args.use_fallback
    )

    # Load checkpoint if provided
    if args.checkpoint and not args.no_checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            print(f"Checkpoint config: {checkpoint['config']}")

    model = model.to(device)

    # Load data
    print(f"Loading data from: {args.data_path}")
    dataset = NavigationDataset(data_path=args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Get sample batch
    sample_tokens, sample_targets = next(iter(dataloader))

    # Run analyses
    analyze_model_structure(model)
    analyze_forward_pass(model, sample_tokens, device)

    if args.full:
        analyze_gradient_flow(model, sample_tokens, sample_targets, device)

    analyze_predictions(model, dataloader, device)

    print("\n" + "="*70)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
