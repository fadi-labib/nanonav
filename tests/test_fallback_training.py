import os

import pytest

from trm_nav.dataset import build_dataset, save_dataset
from trm_nav.train import train


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


run_official = os.environ.get("RUN_OFFICIAL_TRM_TESTS") == "1"


@pytest.mark.skipif(not run_official, reason="Set RUN_OFFICIAL_TRM_TESTS=1 to run official TRM comparison (can be unstable on CPU).")
@pytest.mark.xfail(reason="Official TRM path currently detaches gradients; expected to fail until fixed.")
def test_official_trm_plateaus_without_gradient_fix(tmp_path):
    """
    Comparison check: the official TRM path (use_fallback=False) should learn once gradient
    flow is fixed. Until then, it is expected to stay near random and this test is marked xfail.
    """
    train_data = build_dataset(num_samples=256, grid_size=4, obstacle_density=0.1, base_seed=123)
    val_data = build_dataset(num_samples=64, grid_size=4, obstacle_density=0.1, base_seed=999)

    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    save_dataset(train_data, train_path)
    save_dataset(val_data, val_path)

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
        use_fallback=False,
    )

    max_val_acc = max(history["val_acc"])
    assert max_val_acc >= 0.35, f"Official TRM path did not learn (max val acc {max_val_acc:.3f})"
