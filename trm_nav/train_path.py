"""
Path Prediction Training Loop

Trains TRM to predict the full path (sequence-to-sequence), giving
64x more gradient signal per sample compared to single action prediction.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import sys

from .dataset import PathPredictionDataset, build_path_dataset, save_dataset
from .model import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, ignore_index=self.ignore_index, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    # Track path-specific metrics
    path_correct = 0  # True positives
    path_total = 0    # Total actual path cells

    use_amp = scaler is not None

    for tokens, labels in tqdm(dataloader, desc="Train", leave=False):
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision forward pass
        with autocast('cuda', enabled=use_amp):
            logits = model(tokens)
            loss = criterion(logits.view(-1, 4), labels.view(-1))

        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * tokens.size(0)

        # Accuracy: only count non-ignored tokens (labels != -100)
        preds = logits.argmax(dim=-1)  # (batch, seq_len)
        mask = labels != -100
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()

        # Path recall: what % of actual path cells (class 3) are correctly predicted
        path_mask = labels == 3
        path_correct += ((preds == 3) & path_mask).sum().item()
        path_total += path_mask.sum().item()

    path_recall = path_correct / path_total if path_total > 0 else 0

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': correct / total if total > 0 else 0,
        'path_recall': path_recall
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    path_correct = 0
    path_total = 0

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Val", leave=False):
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens)
            loss = criterion(logits.view(-1, 4), labels.view(-1))

            total_loss += loss.item() * tokens.size(0)

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

            path_mask = labels == 3
            path_correct += ((preds == 3) & path_mask).sum().item()
            path_total += path_mask.sum().item()

    path_recall = path_correct / path_total if path_total > 0 else 0

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': correct / total if total > 0 else 0,
        'path_recall': path_recall
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
    l_cycles: int = 1,  # Inner loop iterations (original TRM uses 4-6)
    batch_size: int = 512,  # Increased for better GPU utilization
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    warmup_epochs: int = 5,  # LR warmup epochs (like original TRM)
    device: str = None,
    num_workers: int = 4,  # Parallel data loading
    use_amp: bool = True,  # Mixed precision training
    use_focal_loss: bool = False,  # Use focal loss instead of CE
    focal_gamma: float = 2.0,  # Focal loss focusing parameter
    grad_last_only: bool = False,  # Like original TRM: only last H-cycle has gradients
):
    """Train path prediction model."""

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
    if use_cuda:
        torch.set_float32_matmul_precision('high')

    print(f"✓ Device: {device}")

    # Load datasets with optimized DataLoader settings
    train_dataset = PathPredictionDataset(data_path=train_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0 and use_cuda,
    )

    val_dataset = PathPredictionDataset(data_path=val_path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0 and use_cuda,
    )

    print(f"✓ Data: {len(train_dataset):,} train, {len(val_dataset):,} val")

    # Create model in path_prediction mode
    model = create_model(
        grid_size=grid_size,
        dim=dim,
        depth=depth,
        dropout=dropout,
        max_recursion_steps=max_recursion_steps,
        l_cycles=l_cycles,
        mode="path_prediction",
        grad_last_only=grad_last_only,
    ).to(device)

    # torch.compile for faster execution (like original TRM)
    if use_cuda and hasattr(torch, 'compile'):
        model = torch.compile(model)
        compiled = True
    else:
        compiled = False

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {num_params:,} params (path prediction mode)")
    print(f"✓ Config: lr={lr}, batch={batch_size}, H_cycles={max_recursion_steps}, L_cycles={l_cycles}")
    print(f"✓ AMP: {'enabled' if use_amp and use_cuda else 'disabled'}, workers={num_workers}, compile={compiled}")
    print(f"✓ Grad: {'last step only' if grad_last_only else 'all steps'}")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + Cosine decay (like original TRM)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # Decay to 10% of max

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler('cuda') if use_amp and use_cuda else None

    # 4-class prediction like original TRM: 0=pad, 1=free, 2=obstacle, 3=path
    # Weight class 3 (path) higher since it's the minority we care about
    class_weights = torch.tensor([0.1, 1.0, 1.0, 8.0], device=device)  # [pad, free, obstacle, path]
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, ignore_index=-100)
        print(f"✓ Loss: Focal (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print(f"✓ Loss: CrossEntropy")

    # Checkpointing
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training...\n")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
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
        print(f"E {epoch+1:3d} │ train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} path={train_metrics['path_recall']:.3f} │ "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} path={val_metrics['path_recall']:.3f} {status}")

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
    parser.add_argument("--max-recursion", type=int, default=8,
                        help="H_cycles: outer loop iterations (default: 8)")
    parser.add_argument("--l-cycles", type=int, default=1,
                        help="L_cycles: inner loop iterations (original TRM uses 4-6)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512 for better GPU utilization)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="LR warmup epochs (default: 5)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--grad-last-only", action="store_true",
                        help="Like original TRM: only last H-cycle has gradients")

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
        l_cycles=args.l_cycles,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        grad_last_only=args.grad_last_only,
    )
