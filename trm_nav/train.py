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
    """Train for one epoch with error handling."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    try:
        # Add progress bar for training batches
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                   desc="Training", leave=False, ncols=100)
        
        for batch_idx, (tokens, actions) in pbar:
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
                
                # Update progress bar with current metrics
                current_loss = total_loss / total
                current_acc = correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nCUDA OOM at batch {batch_idx}:")
                    print(f"  Batch size: {tokens.size(0)}")
                    print(f"  Tokens shape: {tokens.shape}")
                    if device.type == "cuda":
                        print(f"  Available memory: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
                        print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                    print("  Try reducing batch size or model dimension")
                raise e
                
    except Exception as e:
        print(f"Training epoch failed: {e}")
        raise

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
        # Add progress bar for validation batches
        pbar = tqdm(dataloader, desc="Validating", leave=False, ncols=100)
        
        for tokens, actions in pbar:
            tokens = tokens.to(device)
            actions = actions.to(device)

            logits = model(tokens)
            loss = criterion(logits, actions)

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += tokens.size(0)
            
            # Update progress bar with current metrics
            current_loss = total_loss / total
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

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
    # Setup device with error handling - Force CPU due to CUDA driver issues
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        device = torch.device(device)
        print(f"Training on {device}")
        
        # Test CUDA memory availability
        if device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
            
            # Test GPU operations
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            print("GPU operations test passed")
            
    except Exception as e:
        print(f"ERROR: Device setup failed: {e}")
        print("Falling back to CPU...")
        device = torch.device("cpu")

    # Load datasets
    try:
        train_dataset = NavigationDataset(data_path=train_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        print(f"Loaded training data: {len(train_dataset):,} samples")
        
        val_loader = None
        if val_path and Path(val_path).exists():
            val_dataset = NavigationDataset(data_path=val_path)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            print(f"Loaded validation data: {len(val_dataset):,} samples")
            
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}")
        raise

    # Create model with error handling
    try:
        model = create_model(grid_size=grid_size, dim=dim, depth=depth, dropout=dropout, max_recursion_steps=max_recursion_steps)
        print(f"Created model for grid size {grid_size}x{grid_size}")
        
        # Move to device with memory error handling
        if device.type == "cuda":
            try:
                model = model.to(device)
                # Test with dummy input to catch memory issues early
                dummy_input = torch.randint(1, 10, (1, grid_size*grid_size + 4)).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
                print("✓ CUDA compatibility test passed")
                
            except RuntimeError as cuda_error:
                if "out of memory" in str(cuda_error).lower():
                    print(f"CUDA OUT OF MEMORY: {cuda_error}")
                    print("Suggestions:")
                    print(f"  - Reduce batch size (current: {batch_size})")
                    print(f"  - Reduce model dimension (current: {dim})")
                    print(f"  - Use CPU: --device cpu")
                    print("Falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
                else:
                    print(f"CUDA ERROR: {cuda_error}")
                    print("Falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
        else:
            model = model.to(device)
            
    except Exception as e:
        print(f"ERROR: Model creation failed: {e}")
        raise

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

    # Training loop with overall progress
    epoch_pbar = tqdm(range(epochs), desc="Overall Training", ncols=120)
    
    for epoch in epoch_pbar:
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

        # Update overall progress bar
        progress_info = {
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_acc': f"{train_metrics['accuracy']:.4f}"
        }
        if val_loader is not None:
            progress_info.update({
                'val_loss': f"{val_loss:.4f}",
                'val_acc': f"{val_acc:.4f}"
            })
        epoch_pbar.set_postfix(progress_info)

        # Log progress (still print for logging/debugging)
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
    parser.add_argument("--max-recursion", type=int, default=30,
                        help="Maximum recursion steps for TRM (default: 30)")
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
        device=args.device
    )


if __name__ == "__main__":
    main()
