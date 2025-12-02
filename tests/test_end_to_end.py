"""
Test 5: End-to-End Pipeline

Run with:
    python tests/test_end_to_end.py
    pytest tests/test_end_to_end.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from trm_nav.map_generator import generate_solvable_map
from trm_nav.a_star import astar_path, path_to_actions, ACTION_NAMES
from trm_nav.dataset import encode_state, build_dataset, NavigationDataset
from trm_nav.model import create_model


def test_full_pipeline():
    """Test the complete pipeline from map to prediction."""
    print("Step 1: Generate map")
    grid, start, goal = generate_solvable_map(size=8, seed=42)
    print(f"  Grid: 8x8, obstacles: {grid.sum()}")
    print(f"  Start: {start}, Goal: {goal}")

    print("\nStep 2: Compute A* path")
    path = astar_path(grid, start, goal)
    actions = path_to_actions(path)
    print(f"  Path length: {len(path)}")
    print(f"  First 3 actions: {[ACTION_NAMES[a] for a in actions[:3]]}")

    print("\nStep 3: Encode state")
    tokens = encode_state(grid, start, goal)
    print(f"  Token shape: {tokens.shape}")

    print("\nStep 4: Model prediction")
    model = create_model(grid_size=8)
    output = model(tokens.unsqueeze(0))
    pred = output.argmax(dim=-1).item()
    print(f"  Logits: {output[0].detach().numpy().round(2)}")
    print(f"  Predicted: {ACTION_NAMES[pred]} (untrained)")

    print("\n✓ Full pipeline works!")


def test_training_step():
    """Test a single training step."""
    print("Step 1: Build small dataset")
    data = build_dataset(num_samples=32, grid_size=8, base_seed=42)
    dataset = NavigationDataset(data=data)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"  Samples: {len(dataset)}, Batches: {len(loader)}")

    print("\nStep 2: Create model and optimizer")
    model = create_model(grid_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStep 3: Training step")
    model.train()
    batch_tokens, batch_actions = next(iter(loader))

    optimizer.zero_grad()
    output = model(batch_tokens)
    loss = criterion(output, batch_actions)
    loss.backward()
    optimizer.step()

    # Compute accuracy
    preds = output.argmax(dim=-1)
    acc = (preds == batch_actions).float().mean().item()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {acc:.2%} (random baseline: 25%)")

    print("\n✓ Training step works!")


def test_inference_loop():
    """Test inference on multiple samples."""
    print("Step 1: Generate test maps")
    test_maps = []
    for i in range(5):
        grid, start, goal = generate_solvable_map(size=8, seed=1000 + i)
        path = astar_path(grid, start, goal)
        test_maps.append((grid, start, goal, path))
    print(f"  Generated {len(test_maps)} test maps")

    print("\nStep 2: Run inference")
    model = create_model(grid_size=8)
    model.eval()

    results = []
    with torch.no_grad():
        for grid, start, goal, optimal_path in test_maps:
            tokens = encode_state(grid, start, goal).unsqueeze(0)
            logits = model(tokens)
            pred_action = logits.argmax(dim=-1).item()

            optimal_action = path_to_actions(optimal_path)[0] if len(optimal_path) > 1 else 4

            results.append({
                'predicted': ACTION_NAMES[pred_action],
                'optimal': ACTION_NAMES[optimal_action],
                'match': pred_action == optimal_action
            })

    print("\n  Results (untrained model):")
    for i, r in enumerate(results):
        match = "✓" if r['match'] else "✗"
        print(f"    Map {i}: pred={r['predicted']:<6} opt={r['optimal']:<6} {match}")

    matches = sum(r['match'] for r in results)
    print(f"\n  Accuracy: {matches}/{len(results)} (random expected)")

    print("\n✓ Inference loop works!")


def test_evaluation_rollout():
    """Test rollout evaluation (simulated navigation)."""
    from trm_nav.a_star import ACTIONS

    print("Step 1: Setup")
    grid, start, goal = generate_solvable_map(size=8, seed=999)
    model = create_model(grid_size=8)
    model.eval()

    print(f"  Grid: 8x8, Start: {start}, Goal: {goal}")
    optimal_path = astar_path(grid, start, goal)
    print(f"  Optimal path length: {len(optimal_path)}")

    print("\nStep 2: Rollout (max 50 steps)")
    current = start
    path_taken = [current]
    max_steps = 50

    with torch.no_grad():
        for step in range(max_steps):
            if current == goal:
                break

            tokens = encode_state(grid, current, goal).unsqueeze(0)
            logits = model(tokens)
            action = logits.argmax(dim=-1).item()

            # Execute action
            dr, dc = ACTIONS[action]
            new_pos = (current[0] + dr, current[1] + dc)

            # Check validity
            if (0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8 and
                grid[new_pos[0], new_pos[1]] == 0):
                current = new_pos

            path_taken.append(current)

    success = current == goal
    print(f"  Steps taken: {len(path_taken) - 1}")
    print(f"  Final position: {current}")
    print(f"  Success: {success} (unlikely without training)")

    print("\n✓ Rollout evaluation works!")


def run_all():
    """Run all tests with visual output."""
    print("\n" + "=" * 50)
    print("TEST 5: End-to-End Pipeline")
    print("=" * 50 + "\n")

    print("-" * 40)
    test_full_pipeline()
    print()

    print("-" * 40)
    test_training_step()
    print()

    print("-" * 40)
    test_inference_loop()
    print()

    print("-" * 40)
    test_evaluation_rollout()
    print()

    print("=" * 50)
    print("All end-to-end tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()
