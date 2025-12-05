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
import sys
import gc
from typing import Optional, Dict

# Disable torch.compile and dynamo to prevent potential memory issues
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

from .dataset import NavigationDataset
from .model import create_model


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    @staticmethod
    def disable():
        """Disable colors for non-TTY output."""
        Colors.RESET = ""
        Colors.BOLD = ""
        Colors.DIM = ""
        Colors.RED = ""
        Colors.GREEN = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.MAGENTA = ""
        Colors.CYAN = ""
        Colors.WHITE = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


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
    device: torch.device,
    pbar: Optional[tqdm] = None
) -> Dict[str, float]:
    """Train for one epoch with error handling."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    update_freq = max(1, num_batches // 10)  # Update ~10 times per epoch

    try:
        for batch_idx, (tokens, actions) in enumerate(dataloader):
            try:
                tokens = tokens.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)

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

                # Update parent progress bar periodically (not every batch)
                if pbar is not None and (batch_idx % update_freq == 0 or batch_idx == num_batches - 1):
                    current_loss = total_loss / total
                    current_acc = correct / total
                    pbar.set_postfix_str(
                        f"{Colors.BLUE}train{Colors.RESET} {batch_idx+1}/{num_batches} "
                        f"loss={current_loss:.4f} acc={current_acc:.3f}"
                    )

            except (RuntimeError, AttributeError) as e:
                error_str = str(e).lower()
                if "out of memory" in error_str:
                    print(f"\n{Colors.RED}CUDA OOM at batch {batch_idx}:{Colors.RESET}")
                    print(f"  Batch size: {tokens.size(0)}")
                    print(f"  Tokens shape: {tokens.shape}")
                    if device.type == "cuda":
                        print(f"  Available memory: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
                        print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                    print("  Try reducing batch size or model dimension")
                    raise e
                elif "'cell' object" in str(e):
                    # Rare PyTorch/CUDA memory corruption - clear cache and retry
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    tqdm.write(f"{Colors.YELLOW}⚠ Memory glitch at batch {batch_idx}, retrying...{Colors.RESET}")
                    # Retry this batch
                    tokens = tokens.to(device, non_blocking=True)
                    actions = actions.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    logits = model(tokens)
                    loss = criterion(logits, actions)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item() * tokens.size(0)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == actions).sum().item()
                    total += tokens.size(0)
                else:
                    raise e

    except Exception as e:
        print(f"{Colors.RED}Training epoch failed: {e}{Colors.RESET}")
        raise

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pbar: Optional[tqdm] = None
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    update_freq = max(1, num_batches // 5)  # Update ~5 times during validation

    with torch.no_grad():
        for batch_idx, (tokens, actions) in enumerate(dataloader):
            tokens = tokens.to(device)
            actions = actions.to(device)

            logits = model(tokens)
            loss = criterion(logits, actions)

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += tokens.size(0)

            # Update parent progress bar periodically
            if pbar is not None and (batch_idx % update_freq == 0 or batch_idx == num_batches - 1):
                current_loss = total_loss / total
                current_acc = correct / total
                pbar.set_postfix_str(
                    f"{Colors.MAGENTA}val{Colors.RESET} {batch_idx+1}/{num_batches} "
                    f"loss={current_loss:.4f} acc={current_acc:.3f}"
                )

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
    max_recursion_steps: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    device: Optional[str] = None,
    save_every: int = 10,
    resume: bool = True,
    use_fallback: bool = False,
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
    # Keep a single source of truth for run configuration (persisted in checkpoints)
    current_config = {
        'train_path': train_path,
        'val_path': val_path,
        'grid_size': grid_size,
        'dim': dim,
        'depth': depth,
        'dropout': dropout,
        'max_recursion_steps': max_recursion_steps,
        'use_fallback': use_fallback,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'epochs': epochs,
        'save_every': save_every,
    }

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(device)
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name()
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            # Quick GPU test
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            print(f"{Colors.GREEN}✓{Colors.RESET} GPU: {Colors.CYAN}{gpu_name}{Colors.RESET} ({gpu_mem:.1f} GB)")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.RESET} Using CPU (no GPU available)")
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.RESET} Device setup failed: {e}, falling back to CPU")
        device = torch.device("cpu")

    # Load datasets
    try:
        train_dataset = NavigationDataset(data_path=train_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_loader = None
        val_size = 0
        if val_path and Path(val_path).exists():
            val_dataset = NavigationDataset(data_path=val_path)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            val_size = len(val_dataset)

        print(f"{Colors.GREEN}✓{Colors.RESET} Data: {Colors.CYAN}{len(train_dataset):,}{Colors.RESET} train"
              + (f", {Colors.CYAN}{val_size:,}{Colors.RESET} val" if val_loader else ""))
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.RESET} Failed to load datasets: {e}")
        raise

    # Create model
    try:
        model = create_model(
            grid_size=grid_size,
            dim=dim,
            depth=depth,
            dropout=dropout,
            max_recursion_steps=max_recursion_steps,
            use_fallback=use_fallback
        )
    except Exception as model_error:
        print(f"{Colors.RED}✗{Colors.RESET} Model creation failed: {model_error}")
        import traceback
        traceback.print_exc()
        return

    # Move to device
    if device.type == "cuda":
        try:
            model = model.to(device)
            # Quick forward pass test
            dummy_input = torch.randint(1, 10, (1, grid_size*grid_size + 4)).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
        except (RuntimeError, Exception) as cuda_error:
            if "out of memory" in str(cuda_error).lower():
                print(f"{Colors.RED}✗{Colors.RESET} CUDA OOM - reduce batch_size ({batch_size}) or dim ({dim})")
            else:
                print(f"{Colors.RED}✗{Colors.RESET} CUDA error: {cuda_error}")
            print(f"  Falling back to CPU...")
            device = torch.device("cpu")
            model = model.to(device)
    else:
        model = model.to(device)

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{Colors.GREEN}✓{Colors.RESET} Model: {Colors.CYAN}{num_params:,}{Colors.RESET} params, "
          f"dim={dim}, depth={depth}, grid={grid_size}x{grid_size}")
    print(f"{Colors.GREEN}✓{Colors.RESET} Config: lr={lr}, batch={batch_size}, dropout={dropout}, "
          f"wd={weight_decay}, patience={patience}")

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
    
    # Check for existing checkpoint to resume from
    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pt"
    start_epoch = 0
    
    if resume and latest_checkpoint.exists():
        print(f"Found existing checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        saved_config = checkpoint.get('config')
        if saved_config is None:
            raise ValueError("Checkpoint missing config; delete it or rerun with --no-resume.")

        # Enforce exact config match before resuming to avoid silent mismatches
        mismatches = []
        for key, expected_value in current_config.items():
            saved_value = saved_config.get(key)
            if saved_value != expected_value:
                mismatches.append((key, saved_value, expected_value))
        if mismatches:
            mismatch_lines = "\n".join(
                f"  {k}: checkpoint={sv!r} vs current={cv!r}" for k, sv, cv in mismatches
            )
            raise ValueError(
                f"Config mismatch between checkpoint and current run:\n{mismatch_lines}\n"
                "Aborting resume. Re-run with matching flags or delete the checkpoint."
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
        early_stopping.counter = checkpoint.get('early_stop_counter', 0)
        early_stopping.best_loss = checkpoint.get('early_stop_best_loss', best_val_loss)
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    # Training loop with single unified progress bar
    print(f"\n{Colors.BOLD}Starting training...{Colors.RESET}\n")

    # Create a single progress bar for all training
    total_epochs = epochs - start_epoch
    epoch_pbar = tqdm(
        range(start_epoch, epochs),
        desc=f"Epoch",
        unit="epoch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  {postfix}"
    )

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1:3d}/{epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, pbar=epoch_pbar)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate
        is_best = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, pbar=epoch_pbar)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

            val_acc = val_metrics['accuracy']
            val_loss = val_metrics['loss']

            # Save best model (by validation loss, not accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                is_best = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': current_config
                }, checkpoint_dir / "best.pt")

            # Check early stopping
            if early_stopping(val_loss):
                epoch_pbar.close()
                print(f"\n{Colors.YELLOW}Early stopping triggered at epoch {epoch + 1}{Colors.RESET}")
                print(f"Best: loss={Colors.GREEN}{best_val_loss:.4f}{Colors.RESET} acc={Colors.GREEN}{best_val_acc:.4f}{Colors.RESET}")
                break
        else:
            val_loss = None
            val_acc = None

        # Save checkpoint after each epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'history': history,
            'early_stop_counter': early_stopping.counter,
            'early_stop_best_loss': early_stopping.best_loss,
            'config': current_config
        }
        torch.save(checkpoint_data, checkpoint_dir / "latest_checkpoint.pt")

        # Update scheduler
        scheduler.step()

        # Print clean epoch summary (on new line after progress bar updates)
        best_marker = f" {Colors.GREEN}★ best{Colors.RESET}" if is_best else ""
        patience_info = f" {Colors.DIM}[es:{early_stopping.counter}/{patience}]{Colors.RESET}" if val_loader else ""

        tqdm.write(
            f"{Colors.BOLD}E{epoch+1:3d}{Colors.RESET} │ "
            f"train: {Colors.BLUE}loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f}{Colors.RESET}"
            + (f" │ val: {Colors.MAGENTA}loss={val_loss:.4f} acc={val_acc:.3f}{Colors.RESET}" if val_loader else "")
            + best_marker + patience_info
        )

        # Periodic cleanup to prevent memory fragmentation
        if (epoch + 1) % 5 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': current_config
                }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': current_config
    }, checkpoint_dir / "final.pt")

    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{Colors.BOLD}{Colors.GREEN}Training complete!{Colors.RESET}")
    print(f"Best validation: loss={Colors.GREEN}{best_val_loss:.4f}{Colors.RESET} acc={Colors.GREEN}{best_val_acc:.4f}{Colors.RESET}")
    print(f"Checkpoints saved to: {Colors.CYAN}{checkpoint_dir}{Colors.RESET}")
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
    parser.add_argument("--max-recursion", type=int, default=30,
                        help="Maximum recursion steps for TRM (default: 30)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from existing checkpoint")
    parser.add_argument("--use-fallback", action="store_true",
                        help="Use simple transformer fallback instead of official TRM (for debugging)")

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
        resume=not args.no_resume,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        use_fallback=args.use_fallback
    )


if __name__ == "__main__":
    main()
