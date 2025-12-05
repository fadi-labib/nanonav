import os

import pytest
import torch

from trm_nav.dataset import build_dataset, save_dataset
from trm_nav.train import train
from trm_nav.model import create_model


def test_fallback_training_improves_accuracy(tmp_path):
    """
    Sanity check: fallback MLP should learn above random (~25%) within a few epochs.
    This guards against gradient detachment or pipeline breakage in the fast path.
    """
    # Small deterministic datasets for speed
    train_data = build_dataset(num_samples=256, grid_size=4, obstacle_density=0.1, base_seed=123)
    val_data = build_dataset(num_samples=64, grid_size=4, obstacle_density=0.1, base_seed=999)

    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    save_dataset(train_data, train_path)
    save_dataset(val_data, val_path)

    # Train with the lightweight fallback model; keep epochs small for test runtime
    history = train(
        train_path=str(train_path),
        val_path=str(val_path),
        checkpoint_dir=tmp_path / "ckpts",
        grid_size=4,
        dim=16,
        depth=1,
        dropout=0.0,
        max_recursion_steps=4,
        batch_size=64,
        lr=5e-3,
        weight_decay=0.0,
        epochs=10,
        patience=5,
        resume=False,
        device="cpu",
        use_fallback=True,
    )

    max_val_acc = max(history["val_acc"])
    # Expect better than random (25% for 4 actions); 0.35 leaves headroom for noise.
    assert max_val_acc >= 0.35, f"Fallback model failed to learn (max val acc {max_val_acc:.3f})"


def test_official_trm_backprop_flows():
    """
    Ensure the official TRM path (use_fallback=False) propagates gradients into the backbone.
    This is the regression guard against the previous hidden-state detachment bug.
    """
    model = create_model(grid_size=4, dim=16, max_recursion_steps=4, use_fallback=False)
    model.train()

    tokens = torch.randint(0, 10, (2, 4 * 4 + 4))
    targets = torch.tensor([0, 1])
    loss = torch.nn.CrossEntropyLoss()(model(tokens), targets)
    loss.backward()

    # Backbone params should receive gradients (embedding or attention weights)
    has_backbone_grad = any(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for name, p in model.named_parameters()
        if "inner.embed_tokens" in name or "self_attn" in name
    )
    assert has_backbone_grad, "Official TRM backbone did not receive gradients"
