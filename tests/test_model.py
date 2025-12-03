"""
Test 4: Model Forward Pass

Run with:
    python tests/test_model.py
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from trm_nav.model import create_model, TRMNavigator
from trm_nav.dataset import encode_state
from trm_nav.a_star import ACTION_NAMES


def test_trm_available():
    """Check if TRM library is available (required - no fallback)."""
    # If we get here, TRM is available (import would fail otherwise)
    print("✓ tiny-recursive-model available: True (required)")


def test_model_creation():
    """Model creates with correct config."""
    model = create_model(grid_size=8, dim=64, depth=2)

    assert isinstance(model, TRMNavigator)
    assert model.dim == 64
    assert model.seq_len == 68  # 8*8 + 4
    assert model.num_actions == 5

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")


def test_forward_pass_shape():
    """Forward pass produces correct output shape."""
    model = create_model(grid_size=8)

    batch_size = 4
    seq_len = 68
    tokens = torch.randint(0, 10, (batch_size, seq_len))

    output = model(tokens)

    assert output.shape == (batch_size, 5), f"Expected (4, 5), got {output.shape}"
    print(f"✓ Forward pass: input {tuple(tokens.shape)} → output {tuple(output.shape)}")


def test_single_sample():
    """Model handles single sample."""
    model = create_model(grid_size=8)

    tokens = torch.randint(0, 10, (1, 68))
    output = model(tokens)

    assert output.shape == (1, 5)
    print(f"✓ Single sample: output shape {tuple(output.shape)}")


def test_output_is_logits():
    """Output are logits (can be any real number)."""
    model = create_model(grid_size=8)
    tokens = torch.randint(0, 10, (2, 68))

    output = model(tokens)

    # Logits can be positive or negative
    assert output.dtype == torch.float32, "Output should be float"

    # Softmax should sum to 1
    probs = torch.softmax(output, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2)), "Softmax should sum to 1"

    print(f"✓ Output logits: range [{output.min():.2f}, {output.max():.2f}]")


def test_predict_action():
    """predict_action returns valid action indices."""
    model = create_model(grid_size=8)
    tokens = torch.randint(0, 10, (3, 68))

    actions = model.predict_action(tokens)

    assert actions.shape == (3,), f"Expected shape (3,), got {actions.shape}"
    assert all(0 <= a <= 4 for a in actions), "Actions should be in [0, 4]"

    print(f"✓ Predicted actions: {actions.tolist()}")


def test_predict_action_probs():
    """predict_action_probs returns valid probabilities."""
    model = create_model(grid_size=8)
    tokens = torch.randint(0, 10, (2, 68))

    probs = model.predict_action_probs(tokens)

    assert probs.shape == (2, 5)
    assert torch.all(probs >= 0), "Probabilities should be >= 0"
    assert torch.all(probs <= 1), "Probabilities should be <= 1"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2)), "Should sum to 1"

    print(f"✓ Action probabilities: {probs[0].detach().numpy().round(3)}")


def test_with_real_state():
    """Model works with real encoded state."""
    model = create_model(grid_size=8)

    # Create a real grid
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3, 3] = 1
    grid[3, 4] = 1
    start = (0, 0)
    goal = (7, 7)

    tokens = encode_state(grid, start, goal).unsqueeze(0)
    output = model(tokens)
    pred_action = output.argmax(dim=-1).item()

    print(f"✓ Real state test:")
    print(f"    Grid: 8x8 with 2 obstacles")
    print(f"    Start: {start}, Goal: {goal}")
    print(f"    Predicted: {ACTION_NAMES[pred_action]} (untrained)")


def test_return_features():
    """Model can return intermediate features."""
    model = create_model(grid_size=8, dim=64)
    tokens = torch.randint(0, 10, (2, 68))

    logits, features = model(tokens, return_features=True)

    assert logits.shape == (2, 5)
    assert features.shape == (2, 64), f"Expected (2, 64), got {features.shape}"

    print(f"✓ Return features: {tuple(features.shape)}")


def test_gradient_flow():
    """Gradients flow through the model."""
    model = create_model(grid_size=8)
    tokens = torch.randint(0, 10, (2, 68))
    targets = torch.tensor([0, 1])

    output = model(tokens)
    loss = torch.nn.functional.cross_entropy(output, targets)
    loss.backward()

    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "Model should have gradients after backward"
    print(f"✓ Gradient flow: loss={loss.item():.4f}")


def run_all():
    """Run all tests with visual output."""
    print("\n" + "=" * 50)
    print("TEST 4: Model Forward Pass")
    print("=" * 50 + "\n")

    test_trm_available()
    test_model_creation()
    test_forward_pass_shape()
    test_single_sample()
    test_output_is_logits()
    test_predict_action()
    test_predict_action_probs()
    test_with_real_state()
    test_return_features()
    test_gradient_flow()

    print("\n" + "-" * 50)
    print("All model tests passed!")
    print("-" * 50)


if __name__ == "__main__":
    run_all()
