"""
TRM Debug Module

Comprehensive debugging utilities for investigating model internals.
Enable with --debug flag during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


@dataclass
class DebugStats:
    """Container for debug statistics collected during training."""

    # Gradient statistics per layer
    gradient_norms: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gradient_means: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gradient_maxs: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Hidden state statistics across recursion
    hidden_state_norms: List[List[float]] = field(default_factory=list)  # [batch][recursion_step]
    hidden_state_changes: List[List[float]] = field(default_factory=list)  # Change per recursion
    z_H_stats: List[Dict[str, float]] = field(default_factory=list)
    z_L_stats: List[Dict[str, float]] = field(default_factory=list)

    # Output statistics
    logit_stats: List[Dict[str, float]] = field(default_factory=list)
    prediction_entropy: List[float] = field(default_factory=list)
    class_distribution: List[List[int]] = field(default_factory=list)

    # Loss components
    loss_values: List[float] = field(default_factory=list)

    # Per-epoch aggregates
    epoch_summaries: List[Dict[str, Any]] = field(default_factory=list)


class GradientMonitor:
    """Hook-based gradient monitoring."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_hooks(self):
        """Register backward hooks on all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._save_gradient(n, grad)
                )
                self.hooks.append(hook)

    def _save_gradient(self, name: str, grad: torch.Tensor):
        """Save gradient for analysis."""
        if grad is not None:
            self.gradients[name] = grad.detach().clone()

    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all captured gradients."""
        stats = {}
        for name, grad in self.gradients.items():
            if grad is not None:
                stats[name] = {
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                }
        return stats

    def clear(self):
        """Clear captured gradients."""
        self.gradients.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class TRMDebugger:
    """Main debugger class for TRM investigation."""

    def __init__(self, model: nn.Module, enabled: bool = True, output_dir: str = "debug_output"):
        self.model = model
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = DebugStats()
        self.gradient_monitor = GradientMonitor(model) if enabled else None
        self.current_epoch = 0
        self.batch_count = 0

        # Track recursion internals
        self._recursion_states: List[torch.Tensor] = []
        self._original_forward = None

        if enabled:
            self._setup_hooks()

    def _setup_hooks(self):
        """Setup hooks to capture internal states."""
        if self.gradient_monitor:
            self.gradient_monitor.register_hooks()

    def capture_hidden_states(self, z_H: torch.Tensor, z_L: torch.Tensor, step: int):
        """Capture hidden state statistics at a recursion step."""
        if not self.enabled:
            return

        self._recursion_states.append({
            'step': step,
            'z_H_norm': z_H.norm().item(),
            'z_H_mean': z_H.mean().item(),
            'z_H_std': z_H.std().item(),
            'z_L_norm': z_L.norm().item(),
            'z_L_mean': z_L.mean().item(),
            'z_L_std': z_L.std().item(),
        })

    def on_forward_start(self, tokens: torch.Tensor):
        """Called at the start of forward pass."""
        if not self.enabled:
            return
        self._recursion_states = []
        self.batch_count += 1

    def on_forward_end(self, logits: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Called at the end of forward pass."""
        if not self.enabled:
            return

        # Compute logit statistics
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

            self.stats.logit_stats.append({
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'max': logits.max().item(),
                'min': logits.min().item(),
            })
            self.stats.prediction_entropy.append(entropy.item())

            # Track class distribution
            preds = logits.argmax(dim=-1)
            class_counts = [(preds == i).sum().item() for i in range(logits.shape[-1])]
            self.stats.class_distribution.append(class_counts)

    def on_backward_end(self):
        """Called after backward pass."""
        if not self.enabled or not self.gradient_monitor:
            return

        grad_stats = self.gradient_monitor.get_gradient_stats()

        for name, stats in grad_stats.items():
            self.stats.gradient_norms[name].append(stats['norm'])
            self.stats.gradient_means[name].append(stats['mean'])
            self.stats.gradient_maxs[name].append(stats['max'])

        self.gradient_monitor.clear()

    def on_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Called at end of epoch - generate summary."""
        if not self.enabled:
            return

        self.current_epoch = epoch

        summary = {
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0),
            'train_acc': train_metrics.get('accuracy', 0),
            'val_loss': val_metrics.get('loss', 0) if val_metrics else None,
            'val_acc': val_metrics.get('accuracy', 0) if val_metrics else None,
        }

        # Aggregate gradient stats
        if self.stats.gradient_norms:
            grad_summary = {}
            for name, norms in self.stats.gradient_norms.items():
                if norms:
                    grad_summary[name] = {
                        'mean_norm': np.mean(norms),
                        'max_norm': np.max(norms),
                        'min_norm': np.min(norms),
                    }
            summary['gradient_summary'] = grad_summary

        # Aggregate output stats
        if self.stats.prediction_entropy:
            summary['mean_entropy'] = np.mean(self.stats.prediction_entropy)
            summary['entropy_trend'] = 'stable' if len(self.stats.prediction_entropy) < 2 else (
                'increasing' if self.stats.prediction_entropy[-1] > self.stats.prediction_entropy[0] else 'decreasing'
            )

        # Check for collapse to uniform
        if self.stats.class_distribution:
            recent_dist = self.stats.class_distribution[-100:] if len(self.stats.class_distribution) > 100 else self.stats.class_distribution
            avg_dist = np.mean(recent_dist, axis=0)
            total = sum(avg_dist)
            if total > 0:
                normalized = [c / total for c in avg_dist]
                summary['class_balance'] = normalized
                summary['is_collapsed'] = max(normalized) > 0.9 or abs(max(normalized) - min(normalized)) < 0.05

        self.stats.epoch_summaries.append(summary)

        # Print summary
        self._print_epoch_summary(summary)

    def _print_epoch_summary(self, summary: Dict):
        """Print formatted epoch summary."""
        print("\n" + "="*60)
        print(f"DEBUG SUMMARY - Epoch {summary['epoch']}")
        print("="*60)

        # Check for problems
        problems = []

        # 1. Check gradient flow
        if 'gradient_summary' in summary:
            zero_grad_layers = []
            exploding_layers = []
            for name, stats in summary['gradient_summary'].items():
                if stats['mean_norm'] < 1e-8:
                    zero_grad_layers.append(name)
                if stats['max_norm'] > 100:
                    exploding_layers.append(name)

            if zero_grad_layers:
                problems.append(f"ZERO GRADIENTS in: {', '.join(zero_grad_layers[:3])}...")
            if exploding_layers:
                problems.append(f"EXPLODING GRADIENTS in: {', '.join(exploding_layers[:3])}...")

        # 2. Check entropy (collapse detection)
        if 'mean_entropy' in summary:
            # For 4 classes, max entropy is ln(4) ≈ 1.386
            max_entropy = np.log(4)
            normalized_entropy = summary['mean_entropy'] / max_entropy
            print(f"\nOutput Entropy: {summary['mean_entropy']:.4f} ({normalized_entropy*100:.1f}% of max)")

            if normalized_entropy > 0.95:
                problems.append("OUTPUT COLLAPSED TO UNIFORM (near-random predictions)")
            elif normalized_entropy < 0.3:
                problems.append("OUTPUT COLLAPSED TO SINGLE CLASS")

        # 3. Check class balance
        if 'class_balance' in summary:
            balance = summary['class_balance']
            print(f"Class Distribution: {[f'{b:.2%}' for b in balance]}")
            if summary.get('is_collapsed'):
                problems.append(f"CLASS IMBALANCE DETECTED")

        # 4. Accuracy check
        train_acc = summary.get('train_acc', 0)
        val_acc = summary.get('val_acc')

        print(f"\nAccuracy: train={train_acc:.4f}", end="")
        if val_acc is not None:
            print(f", val={val_acc:.4f}")
            if abs(train_acc - 0.25) < 0.05 and abs(val_acc - 0.25) < 0.05:
                problems.append("ACCURACY AT RANDOM CHANCE (0.25) - Model not learning!")
        else:
            print()

        # Print gradient summary (top layers)
        if 'gradient_summary' in summary:
            print("\nGradient Norms (selected layers):")
            # Sort by norm and show extremes
            sorted_grads = sorted(
                summary['gradient_summary'].items(),
                key=lambda x: x[1]['mean_norm'],
                reverse=True
            )

            # Show top 5 and bottom 5
            print("  Highest:")
            for name, stats in sorted_grads[:5]:
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
                print(f"    {short_name}: {stats['mean_norm']:.2e}")

            print("  Lowest:")
            for name, stats in sorted_grads[-5:]:
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
                print(f"    {short_name}: {stats['mean_norm']:.2e}")

        # Print problems
        if problems:
            print("\n" + "!"*60)
            print("POTENTIAL ISSUES DETECTED:")
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}")
            print("!"*60)
        else:
            print("\nNo obvious issues detected.")

        print("="*60 + "\n")

    def generate_plots(self, save_path: Optional[str] = None):
        """Generate diagnostic plots."""
        if not self.enabled:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('TRM Debug Diagnostics', fontsize=14)

        # 1. Gradient norms over time (top layers)
        ax = axes[0, 0]
        if self.stats.gradient_norms:
            # Get embedding and output layer gradients
            key_layers = []
            for name in self.stats.gradient_norms.keys():
                if 'embed' in name.lower() or 'classifier' in name.lower() or 'lm_head' in name.lower():
                    key_layers.append(name)

            if not key_layers:
                key_layers = list(self.stats.gradient_norms.keys())[:5]

            for name in key_layers[:5]:
                norms = self.stats.gradient_norms[name]
                if norms:
                    short_name = '.'.join(name.split('.')[-2:])
                    ax.plot(norms, label=short_name, alpha=0.7)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norms (Key Layers)')
            ax.legend(fontsize=8)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No gradient data', ha='center', va='center')

        # 2. Prediction entropy over time
        ax = axes[0, 1]
        if self.stats.prediction_entropy:
            ax.plot(self.stats.prediction_entropy, alpha=0.7)
            ax.axhline(y=np.log(4), color='r', linestyle='--', label='Max entropy (uniform)')
            ax.axhline(y=0, color='g', linestyle='--', label='Min entropy (certain)')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Entropy')
            ax.set_title('Prediction Entropy')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center')

        # 3. Class distribution over time
        ax = axes[0, 2]
        if self.stats.class_distribution:
            dist_array = np.array(self.stats.class_distribution)
            # Normalize
            row_sums = dist_array.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            dist_normalized = dist_array / row_sums

            for i in range(dist_normalized.shape[1]):
                ax.plot(dist_normalized[:, i], label=f'Class {i}', alpha=0.7)
            ax.axhline(y=0.25, color='gray', linestyle='--', label='Uniform (0.25)')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Proportion')
            ax.set_title('Predicted Class Distribution')
            ax.legend()
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No class distribution data', ha='center', va='center')

        # 4. Logit statistics
        ax = axes[1, 0]
        if self.stats.logit_stats:
            means = [s['mean'] for s in self.stats.logit_stats]
            stds = [s['std'] for s in self.stats.logit_stats]
            ax.plot(means, label='Mean', alpha=0.7)
            ax.fill_between(range(len(means)),
                           [m-s for m, s in zip(means, stds)],
                           [m+s for m, s in zip(means, stds)],
                           alpha=0.3, label='±1 std')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Logit Value')
            ax.set_title('Logit Statistics')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No logit data', ha='center', va='center')

        # 5. Training curve (from epoch summaries)
        ax = axes[1, 1]
        if self.stats.epoch_summaries:
            epochs = [s['epoch'] for s in self.stats.epoch_summaries]
            train_acc = [s['train_acc'] for s in self.stats.epoch_summaries]
            val_acc = [s['val_acc'] for s in self.stats.epoch_summaries if s['val_acc'] is not None]

            ax.plot(epochs, train_acc, label='Train Acc', marker='o')
            if val_acc:
                ax.plot(epochs[:len(val_acc)], val_acc, label='Val Acc', marker='s')
            ax.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training Progress')
            ax.legend()
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No epoch data', ha='center', va='center')

        # 6. Gradient flow heatmap
        ax = axes[1, 2]
        if self.stats.gradient_norms and self.stats.epoch_summaries:
            # Create heatmap of gradient norms across layers
            layer_names = list(self.stats.gradient_norms.keys())[:20]  # Top 20 layers

            if layer_names:
                # Aggregate by epoch (take mean of batches within epoch)
                n_epochs = len(self.stats.epoch_summaries)

                if n_epochs > 0:
                    data = []
                    for name in layer_names:
                        norms = self.stats.gradient_norms[name]
                        if norms:
                            # Take last value (most recent)
                            data.append(np.log10(norms[-1] + 1e-10))
                        else:
                            data.append(-10)

                    short_names = ['.'.join(n.split('.')[-2:])[:15] for n in layer_names]

                    ax.barh(range(len(data)), data)
                    ax.set_yticks(range(len(short_names)))
                    ax.set_yticklabels(short_names, fontsize=7)
                    ax.set_xlabel('log10(Gradient Norm)')
                    ax.set_title('Gradient Flow (Latest)')
                    ax.axvline(x=-6, color='r', linestyle='--', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'No epoch data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No gradient layers', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No gradient data', ha='center', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Debug plots saved to: {save_path}")
        else:
            save_path = self.output_dir / f"debug_epoch_{self.current_epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Debug plots saved to: {save_path}")

        plt.close()

    def save_stats(self, filename: str = "debug_stats.json"):
        """Save debug statistics to JSON."""
        if not self.enabled:
            return

        # Convert to serializable format
        save_data = {
            'epoch_summaries': self.stats.epoch_summaries,
            'final_gradient_norms': {
                name: norms[-100:] if len(norms) > 100 else norms
                for name, norms in self.stats.gradient_norms.items()
            },
            'final_entropy': self.stats.prediction_entropy[-100:] if len(self.stats.prediction_entropy) > 100 else self.stats.prediction_entropy,
            'final_class_dist': self.stats.class_distribution[-100:] if len(self.stats.class_distribution) > 100 else self.stats.class_distribution,
        }

        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"Debug stats saved to: {save_path}")

    def print_quick_diagnosis(self):
        """Print a quick diagnosis based on collected stats."""
        if not self.enabled:
            print("Debugger not enabled.")
            return

        print("\n" + "="*60)
        print("QUICK DIAGNOSIS")
        print("="*60)

        issues = []

        # Check 1: Accuracy at random chance
        if self.stats.epoch_summaries:
            last_summary = self.stats.epoch_summaries[-1]
            if last_summary['train_acc'] < 0.30:
                issues.append({
                    'severity': 'CRITICAL',
                    'issue': 'Model accuracy near random chance',
                    'detail': f"Train acc: {last_summary['train_acc']:.4f} (expected >0.30)",
                    'likely_cause': 'Gradients not flowing or model architecture broken',
                    'action': 'Check gradient flow through TRM recursion'
                })

        # Check 2: Zero gradients
        if self.stats.gradient_norms:
            zero_grad_layers = []
            for name, norms in self.stats.gradient_norms.items():
                if norms and np.mean(norms[-10:]) < 1e-8:
                    zero_grad_layers.append(name)

            if zero_grad_layers:
                issues.append({
                    'severity': 'CRITICAL',
                    'issue': f'Zero gradients in {len(zero_grad_layers)} layers',
                    'detail': f"Affected: {', '.join(zero_grad_layers[:3])}...",
                    'likely_cause': 'Detached tensors or torch.no_grad() in forward',
                    'action': 'Check for .detach() calls in TRM forward pass'
                })

        # Check 3: Uniform output
        if self.stats.prediction_entropy:
            recent_entropy = np.mean(self.stats.prediction_entropy[-100:])
            max_entropy = np.log(4)
            if recent_entropy > 0.95 * max_entropy:
                issues.append({
                    'severity': 'HIGH',
                    'issue': 'Model outputting near-uniform distribution',
                    'detail': f"Entropy: {recent_entropy:.4f} (max: {max_entropy:.4f})",
                    'likely_cause': 'Classifier not receiving meaningful features',
                    'action': 'Check that hidden states change across recursion'
                })

        # Check 4: Class collapse
        if self.stats.class_distribution:
            recent_dist = np.mean(self.stats.class_distribution[-100:], axis=0)
            if recent_dist.sum() > 0:
                normalized = recent_dist / recent_dist.sum()
                if max(normalized) > 0.8:
                    issues.append({
                        'severity': 'HIGH',
                        'issue': f'Model collapsed to predicting class {np.argmax(normalized)}',
                        'detail': f"Distribution: {[f'{d:.2%}' for d in normalized]}",
                        'likely_cause': 'Class imbalance or gradient issues',
                        'action': 'Check training data balance and loss function'
                    })

        if not issues:
            print("No critical issues detected based on collected stats.")
            print("Model appears to be training normally.")
        else:
            for i, issue in enumerate(issues, 1):
                print(f"\n[{issue['severity']}] Issue {i}: {issue['issue']}")
                print(f"  Detail: {issue['detail']}")
                print(f"  Likely cause: {issue['likely_cause']}")
                print(f"  Suggested action: {issue['action']}")

        print("\n" + "="*60)

    def cleanup(self):
        """Clean up hooks and resources."""
        if self.gradient_monitor:
            self.gradient_monitor.remove_hooks()


def create_debugger(model: nn.Module, enabled: bool = True, output_dir: str = "debug_output") -> TRMDebugger:
    """Factory function to create a debugger instance."""
    return TRMDebugger(model, enabled=enabled, output_dir=output_dir)
