"""
Navigation-specific wrapper for the official Samsung SAIL Montreal TRM implementation.
Uses the TRM from the git submodule at external/trm.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
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
