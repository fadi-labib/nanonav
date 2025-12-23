"""
Path Prediction Training Loop

Trains TRM to predict the full path (sequence-to-sequence), giving
64x more gradient signal per sample compared to single action prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import sys

from .dataset import PathPredictionDataset, build_path_dataset, save_dataset
from .model import create_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for tokens, labels in tqdm(dataloader, desc="Train", leave=False):
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Output: (batch, seq_len, 2)
        logits = model(tokens)

        # Reshape for cross entropy: (batch * seq_len, 2) vs (batch * seq_len,)
        loss = criterion(logits.view(-1, 2), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * tokens.size(0)

        # Accuracy: only count non-ignored tokens (labels != -100)
        preds = logits.argmax(dim=-1)  # (batch, seq_len)
        mask = labels != -100
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': correct / total if total > 0 else 0
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Val", leave=False):
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens)
            loss = criterion(logits.view(-1, 2), labels.view(-1))

            total_loss += loss.item() * tokens.size(0)

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': correct / total if total > 0 else 0
    }


def train(
    train_path: str = "data/train_path.pt",
    val_path: str = "data/test_path.pt",
    checkpoint_dir: str = "checkpoints_path",
    grid_size: int = 8,
    dim: int = 64,
    depth: int = 2,
    dropout: float = 0.1,
    max_recursion_steps: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    device: str = None,
):
    """Train path prediction model."""

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"✓ Device: {device}")

    # Load datasets
    train_dataset = PathPredictionDataset(data_path=train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PathPredictionDataset(data_path=val_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ Data: {len(train_dataset):,} train, {len(val_dataset):,} val")

    # Create model in path_prediction mode
    model = create_model(
        grid_size=grid_size,
        dim=dim,
        depth=depth,
        dropout=dropout,
        max_recursion_steps=max_recursion_steps,
        mode="path_prediction"
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {num_params:,} params (path prediction mode)")
    print(f"✓ Config: lr={lr}, batch={batch_size}, max_recursion={max_recursion_steps}")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore coordinate tokens

    # Checkpointing
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training...\n")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
            }, checkpoint_dir / "best.pt")
        else:
            patience_counter += 1

        status = "★ best" if is_best else f"[es:{patience_counter}/{patience}]"
        print(f"E {epoch+1:3d} │ train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} │ "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} {status}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TRM Path Prediction")
    parser.add_argument("--train-path", default="data/train_path.pt")
    parser.add_argument("--val-path", default="data/test_path.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints_path")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-recursion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoint_dir=args.checkpoint_dir,
        grid_size=args.grid_size,
        dim=args.dim,
        depth=args.depth,
        dropout=args.dropout,
        max_recursion_steps=args.max_recursion,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )
