"""
Navigation-specific wrapper for the official Samsung SAIL Montreal TRM implementation.
Uses the TRM from the git submodule at external/trm.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
import sys
import os
import warnings

# Add the submodule to Python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_trm_path = os.path.join(_project_root, 'external', 'trm')
if _trm_path not in sys.path:
    sys.path.insert(0, _trm_path)

try:
    # Import from the official TRM submodule
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    from models.common import trunc_normal_init_
    from models.layers import CastedEmbedding, CastedLinear
    _TRM_AVAILABLE = True
except ImportError as e:
    _TRM_AVAILABLE = False
    _IMPORT_ERROR = str(e)


@dataclass
class RecursionDebugInfo:
    """Debug information captured during TRM recursion."""
    # Per-step statistics
    step_z_H_norms: List[float] = field(default_factory=list)
    step_z_L_norms: List[float] = field(default_factory=list)
    step_z_H_means: List[float] = field(default_factory=list)
    step_z_L_means: List[float] = field(default_factory=list)
    step_z_H_stds: List[float] = field(default_factory=list)
    step_z_L_stds: List[float] = field(default_factory=list)
    step_changes: List[float] = field(default_factory=list)  # How much z_H changes per step

    # Final output statistics
    final_z_H_norm: float = 0.0
    final_z_H_mean: float = 0.0
    final_z_H_std: float = 0.0
    pooled_features_norm: float = 0.0
    pooled_features_mean: float = 0.0
    pooled_features_std: float = 0.0

    # Classifier input/output
    classifier_input_norm: float = 0.0
    logits_mean: float = 0.0
    logits_std: float = 0.0
    logits_range: float = 0.0  # max - min (indicates confidence)

    # Embedding statistics
    embedding_norm: float = 0.0
    embedding_mean: float = 0.0

    # Gradient info (populated after backward)
    embedding_grad_norm: Optional[float] = None
    classifier_grad_norm: Optional[float] = None


class NavigationTRM(nn.Module):
    """Navigation wrapper for the official TRM implementation."""

    def __init__(
        self,
        seq_len: int = 68,
        vocab_size: int = 256,
        hidden_size: int = 64,
        num_heads: int = 4,
        max_recursion_steps: int = 8,
        dropout: float = 0.1,
        use_fallback: bool = False,
        num_actions: int = 4,
        **kwargs
    ):
        super().__init__()
        self.num_actions = num_actions
        self.use_fallback = use_fallback
        self._debug_mode = False
        self._last_debug_info: Optional[RecursionDebugInfo] = None

        if use_fallback:
            # Always allow fallback regardless of TRM availability
            self._init_fallback(seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions)
        elif not _TRM_AVAILABLE:
            raise ImportError(f"Official TRM not available: {_IMPORT_ERROR}. Please ensure the submodule is properly initialized.")
        else:
            self._init_official(seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions)
    
    def _init_fallback(self, seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions):
        """Fallback implementation using a simple MLP classifier (fast and debuggable)."""
        self.use_official = False
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.max_recursion_steps = max_recursion_steps

        # Simple embedding + MLP (works as a sanity baseline)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)

        mlp_hidden = max(hidden_size * 2, 128)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size * seq_len),
            nn.Linear(hidden_size * seq_len, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_actions)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_actions)
        )

        self._init_weights()
    
    def _init_official(self, seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions):
        """Initialize with official TRM implementation."""
        self.use_official = True
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.max_recursion_steps = max_recursion_steps

        # Create config dict for the official TRM
        config_dict = {
            "batch_size": 1,  # Will be overridden at runtime
            "seq_len": seq_len,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": 1,
            "vocab_size": vocab_size,
            "H_cycles": max_recursion_steps,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 2,
            "hidden_size": hidden_size,
            "expansion": 2.0,
            "num_heads": num_heads,
            "pos_encodings": "rotary",
            "dropout": dropout,
            "halt_max_steps": max_recursion_steps,
            "halt_exploration_prob": 0.0,
            "puzzle_emb_len": 0,  # Explicitly set to 0 since we don't use puzzle embeddings
            # Use full precision for stability (bfloat16 can stall learning on some GPUs/CPUs)
            "forward_dtype": "float32",
        }

        # Initialize the official TRM
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.config = self.trm.config
        self.vocab_size = vocab_size

        # Classification head consumes hidden representations (z_H)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_actions)
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass using official TRM with full z_H/z_L mechanism."""
        if getattr(self, "use_official", True):
            return self._forward_official(tokens)
        return self._forward_fallback(tokens)

    def _forward_official(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward using official TRM forward, pooling gradient-carrying hidden states."""
        batch_size = tokens.shape[0]
        device = tokens.device

        # Update config batch size
        self.config.batch_size = batch_size

        # Build batch and carry
        batch = {
            "inputs": tokens,
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        carry = self.trm.initial_carry(batch)
        carry = self._carry_to_device(carry, device)

        # Run official forward (inner forward already runs all cycles with grad)
        carry_after, outputs = self.trm.forward(carry, batch)

        # Prefer gradient-carrying hidden states
        z_H = outputs.get("last_hidden", carry_after.inner_carry.z_H).float()

        # Pool over all tokens
        pooled_features = z_H.mean(dim=1)

        # Project to action space
        return self.classifier(pooled_features)

    @staticmethod
    def _carry_to_device(carry, device: torch.device):
        """Move the TRM carry tensors to the specified device."""
        inner = carry.inner_carry
        inner = type(inner)(
            z_H=inner.z_H.to(device),
            z_L=inner.z_L.to(device)
        )

        current_data = {k: v.to(device) for k, v in carry.current_data.items()}

        return type(carry)(
            inner_carry=inner,
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data=current_data
        )
    
    def _forward_fallback(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward with simple MLP baseline (no attention)."""
        emb = self.token_embed(tokens)  # (B, seq_len, hidden)
        flat = emb.flatten(start_dim=1)  # (B, seq_len*hidden)
        return self.mlp(flat)

    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode for detailed internal tracking."""
        self._debug_mode = enabled

    def get_last_debug_info(self) -> Optional[RecursionDebugInfo]:
        """Get debug info from the last forward pass (only when debug mode enabled)."""
        return self._last_debug_info

    def forward_with_debug(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, RecursionDebugInfo]:
        """
        Forward pass with detailed debug information about recursion internals.

        This runs a custom forward that captures z_H and z_L at each recursion step
        to help diagnose where the model might be failing.
        """
        if not getattr(self, "use_official", True):
            # Fallback mode - simpler debug
            debug_info = RecursionDebugInfo()
            emb = self.token_embed(tokens)
            debug_info.embedding_norm = emb.norm().item()
            debug_info.embedding_mean = emb.mean().item()
            flat = emb.flatten(start_dim=1)
            logits = self.mlp(flat)
            debug_info.logits_mean = logits.mean().item()
            debug_info.logits_std = logits.std().item()
            debug_info.logits_range = (logits.max() - logits.min()).item()
            self._last_debug_info = debug_info
            return logits, debug_info

        # Official TRM debug forward
        debug_info = RecursionDebugInfo()
        batch_size = tokens.shape[0]
        device = tokens.device

        # Update config batch size
        self.config.batch_size = batch_size

        # Access TRM inner model
        inner = self.trm.inner

        # Get input embeddings
        batch = {
            "inputs": tokens,
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        input_embeddings = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        debug_info.embedding_norm = input_embeddings.norm().item()
        debug_info.embedding_mean = input_embeddings.mean().item()

        # Initialize carry
        seq_info = dict(
            cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
        )

        # Initialize z_H and z_L
        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.seq_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.seq_len, -1).clone()

        # Move to device
        z_H = z_H.to(device)
        z_L = z_L.to(device)

        # Record initial state
        debug_info.step_z_H_norms.append(z_H.norm().item())
        debug_info.step_z_L_norms.append(z_L.norm().item())
        debug_info.step_z_H_means.append(z_H.mean().item())
        debug_info.step_z_L_means.append(z_L.mean().item())
        debug_info.step_z_H_stds.append(z_H.std().item())
        debug_info.step_z_L_stds.append(z_L.std().item())

        # Run recursion steps manually to capture internals
        for h_step in range(self.config.H_cycles):
            z_H_before = z_H.clone()

            # L cycles
            for l_step in range(self.config.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_embeddings, **seq_info)

            # H update
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Record changes
            change = (z_H - z_H_before).norm().item()
            debug_info.step_changes.append(change)
            debug_info.step_z_H_norms.append(z_H.norm().item())
            debug_info.step_z_L_norms.append(z_L.norm().item())
            debug_info.step_z_H_means.append(z_H.mean().item())
            debug_info.step_z_L_means.append(z_L.mean().item())
            debug_info.step_z_H_stds.append(z_H.std().item())
            debug_info.step_z_L_stds.append(z_L.std().item())

        # Final statistics
        debug_info.final_z_H_norm = z_H.norm().item()
        debug_info.final_z_H_mean = z_H.mean().item()
        debug_info.final_z_H_std = z_H.std().item()

        # Pool and classify
        pooled_features = z_H.float().mean(dim=1)
        debug_info.pooled_features_norm = pooled_features.norm().item()
        debug_info.pooled_features_mean = pooled_features.mean().item()
        debug_info.pooled_features_std = pooled_features.std().item()
        debug_info.classifier_input_norm = pooled_features.norm().item()

        logits = self.classifier(pooled_features)
        debug_info.logits_mean = logits.mean().item()
        debug_info.logits_std = logits.std().item()
        debug_info.logits_range = (logits.max() - logits.min()).item()

        self._last_debug_info = debug_info
        return logits, debug_info

    def print_debug_summary(self, debug_info: Optional[RecursionDebugInfo] = None):
        """Print a formatted summary of debug information."""
        info = debug_info or self._last_debug_info
        if info is None:
            print("No debug info available. Run forward_with_debug() first.")
            return

        print("\n" + "="*70)
        print("TRM RECURSION DEBUG SUMMARY")
        print("="*70)

        # Embedding stats
        print(f"\n[INPUT EMBEDDING]")
        print(f"  Norm: {info.embedding_norm:.4f}")
        print(f"  Mean: {info.embedding_mean:.6f}")

        # Recursion analysis
        print(f"\n[RECURSION ANALYSIS] ({len(info.step_changes)} H-cycles)")
        if info.step_changes:
            print(f"  z_H changes per step: {[f'{c:.4f}' for c in info.step_changes[:5]]}{'...' if len(info.step_changes) > 5 else ''}")
            total_change = sum(info.step_changes)
            avg_change = total_change / len(info.step_changes)
            print(f"  Total change: {total_change:.4f}, Avg per step: {avg_change:.4f}")

            if avg_change < 0.001:
                print("  ⚠️  WARNING: z_H barely changing - recursion may be ineffective!")
            elif info.step_changes[-1] < 0.0001:
                print("  ⚠️  WARNING: z_H converged to fixed point - no more refinement")

        # z_H evolution
        print(f"\n[z_H EVOLUTION]")
        print(f"  Initial norm: {info.step_z_H_norms[0]:.4f}")
        print(f"  Final norm:   {info.step_z_H_norms[-1]:.4f}")
        print(f"  Initial std:  {info.step_z_H_stds[0]:.4f}")
        print(f"  Final std:    {info.step_z_H_stds[-1]:.4f}")

        if info.step_z_H_stds[-1] < 0.01:
            print("  ⚠️  WARNING: z_H has collapsed (very low variance)!")

        # Pooled features
        print(f"\n[POOLED FEATURES -> CLASSIFIER]")
        print(f"  Norm: {info.pooled_features_norm:.4f}")
        print(f"  Mean: {info.pooled_features_mean:.6f}")
        print(f"  Std:  {info.pooled_features_std:.4f}")

        # Output analysis
        print(f"\n[OUTPUT LOGITS]")
        print(f"  Mean:  {info.logits_mean:.4f}")
        print(f"  Std:   {info.logits_std:.4f}")
        print(f"  Range: {info.logits_range:.4f}")

        if info.logits_range < 0.1:
            print("  ⚠️  WARNING: Logits have very small range - near-uniform predictions!")
        elif info.logits_std < 0.1:
            print("  ⚠️  WARNING: Low logit variance - model may be underconfident")

        # Gradient info if available
        if info.embedding_grad_norm is not None or info.classifier_grad_norm is not None:
            print(f"\n[GRADIENTS]")
            if info.embedding_grad_norm is not None:
                print(f"  Embedding grad norm: {info.embedding_grad_norm:.6f}")
                if info.embedding_grad_norm < 1e-7:
                    print("  ⚠️  WARNING: Near-zero embedding gradient!")
            if info.classifier_grad_norm is not None:
                print(f"  Classifier grad norm: {info.classifier_grad_norm:.6f}")

        print("\n" + "="*70)

    def predict_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict action (argmax of logits)."""
        logits = self.forward(tokens)
        return logits.argmax(dim=-1)


def create_navigation_model(
    grid_size: int = 8,
    dim: int = 64,
    depth: int = 2,
    max_recursion_steps: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> NavigationTRM:
    """Create a navigation TRM model."""
    seq_len = grid_size * grid_size + 4
    
    return NavigationTRM(
        seq_len=seq_len,
        vocab_size=256,
        hidden_size=dim,
        num_heads=max(1, dim // 16),
        max_recursion_steps=max_recursion_steps,
        dropout=dropout
    )
