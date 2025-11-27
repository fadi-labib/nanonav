"""
Training Loop

Standard PyTorch training for the TRM navigator with regularization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict

from .dataset import NavigationDataset
from .model import create_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for tokens, actions in dataloader:
        tokens = tokens.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, actions)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * tokens.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total += tokens.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, actions in dataloader:
            tokens = tokens.to(device)
            actions = actions.to(device)

            logits = model(tokens)
            loss = criterion(logits, actions)

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += tokens.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def train(
    train_path: str = "data/train.pt",
    val_path: Optional[str] = "data/test.pt",
    checkpoint_dir: str = "checkpoints",
    grid_size: int = 8,
    dim: int = 64,
    depth: int = 2,
    dropout: float = 0.1,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    device: Optional[str] = None,
    save_every: int = 10,
) -> Dict[str, list]:
    """
    Train the TRM navigator model.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        checkpoint_dir: Directory for saving checkpoints
        grid_size: Size of grid (for model config)
        dim: Model dimension
        depth: Number of mixer layers
        dropout: Dropout rate for regularization
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization strength
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: Device to train on
        save_every: Save checkpoint every N epochs

    Returns:
        Dictionary of training history
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on {device}")

    # Load datasets
    train_dataset = NavigationDataset(data_path=train_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = NavigationDataset(data_path=val_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    # Create model with dropout
    model = create_model(grid_size=grid_size, dim=dim, depth=depth, dropout=dropout)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}")
    print(f"Early stopping: patience={patience}")

    # Setup training with weight decay (AdamW)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_val_loss = float('inf')
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

            val_acc = val_metrics['accuracy']
            val_loss = val_metrics['loss']
            val_str = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

            # Save best model (by validation loss, not accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': {
                        'grid_size': grid_size,
                        'dim': dim,
                        'depth': depth,
                        'dropout': dropout
                    }
                }, checkpoint_dir / "best.pt")
                val_str += " *"  # Mark best

            # Check early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}")
                break
        else:
            val_str = ""

        # Update scheduler
        scheduler.step()

        # Log progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f} "
              f"{val_str}")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'grid_size': grid_size,
                    'dim': dim,
                    'depth': depth,
                    'dropout': dropout
                }
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': {
            'grid_size': grid_size,
            'dim': dim,
            'depth': depth,
            'dropout': dropout
        }
    }, checkpoint_dir / "final.pt")

    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    return history


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TRM Navigator")
    parser.add_argument("--train-path", default="data/train.pt")
    parser.add_argument("--val-path", default="data/test.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="L2 regularization (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
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
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device
    )


if __name__ == "__main__":
    main()
