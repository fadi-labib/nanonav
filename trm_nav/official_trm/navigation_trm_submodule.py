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
        num_actions: int = 4,
        **kwargs
    ):
        super().__init__()
        self.num_actions = num_actions

        if not _TRM_AVAILABLE:
            raise ImportError(f"Official TRM not available: {_IMPORT_ERROR}. Please ensure the submodule is properly initialized.")
        else:
            self._init_official(seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions)
    
    def _init_fallback(self, seq_len, vocab_size, hidden_size, num_heads, max_recursion_steps, dropout, num_actions):
        """Fallback implementation using simple transformer."""
        self.use_official = False
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.max_recursion_steps = max_recursion_steps

        # Simple transformer-based implementation
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(seq_len, hidden_size)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(2)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, int(hidden_size * 2.0)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_size * 2.0), hidden_size)
            ) for _ in range(2)
        ])

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
            "puzzle_emb_len": 0  # Explicitly set to 0 since we don't use puzzle embeddings
        }

        # Initialize the official TRM
        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.config = self.trm.config
        self.vocab_size = vocab_size

        # Classification head (hidden states are already in hidden_size dimension)
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
        """Forward pass using official TRM."""
        return self._forward_official(tokens)
    
    def _forward_official(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward with official TRM."""
        batch_size = tokens.shape[0]
        device = tokens.device
        
        # Update config batch size
        self.config.batch_size = batch_size
        
        # Create puzzle identifiers
        puzzle_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Create batch dict for TRM
        batch = {
            "inputs": tokens,
            "puzzle_identifiers": puzzle_ids
        }
        
        # Initialize carry state using official TRM approach
        with torch.device(device):
            carry = self.trm.initial_carry(batch)
        
        # Run forward pass
        carry_after, output = self.trm.forward(carry, batch)

        # Extract HIDDEN STATES (not logits!) from the TRM
        # z_H = high-level reasoning state, shape (batch, seq_len, hidden_size)
        # This is the actual learned representation, NOT next-token predictions
        hidden = carry_after.inner_carry.z_H  # (batch, seq_len, hidden_size)

        # Use LAST 4 tokens (coordinates: start_row, start_col, goal_row, goal_col)
        # These have "seen" the entire grid via attention and contain
        # the most relevant information for deciding the action
        coord_hidden = hidden[:, -4:, :].float()  # (batch, 4, hidden_size)
        pooled = coord_hidden.mean(dim=1)  # (batch, hidden_size)

        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def _forward_fallback(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward with fallback transformer."""
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Embed tokens and positions
        x = self.token_embed(tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)
        
        # Recursive reasoning
        for cycle in range(self.max_recursion_steps):
            for attn, ln, ffn in zip(self.attention_layers, self.layer_norms, self.ffns):
                attn_out, _ = attn(x, x, x)
                x = ln(x + attn_out)
                ffn_out = ffn(x)
                x = ln(x + ffn_out)
        
        # Pool and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
    
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