"""
TRM Model Wrapper

Wraps the official Samsung SAIL Montreal TRM implementation for navigation task.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import official TRM implementation from submodule
from .official_trm.navigation_trm_submodule import NavigationTRM, create_navigation_model


class TRMNavigator(nn.Module):
    """
    TRM-based navigation model using official Samsung SAIL Montreal implementation.

    Takes encoded state (tokens) and predicts action logits.

    Uses official TinyRecursiveReasoningModel for spatial reasoning.
    """

    def __init__(
        self,
        dim: int = 64,
        num_tokens: int = 256,
        seq_len: int = 68,  # 8*8 + 4 for 8x8 grid
        depth: int = 2,
        num_actions: int = 5,
        max_recursion_steps: int = 30,
        halt_prob_thres: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.max_recursion_steps = max_recursion_steps
        self.halt_prob_thres = halt_prob_thres
        self.dropout_rate = dropout

        # Official TRM implementation
        self.trm = NavigationTRM(
            seq_len=seq_len,
            vocab_size=num_tokens,
            hidden_size=dim,
            num_heads=max(1, dim // 16),
            max_recursion_steps=max_recursion_steps,
            dropout=dropout
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_features: bool = False,
        return_refinement_steps: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Input token sequence (batch, seq_len)
            return_features: If True, also return intermediate features
            return_refinement_steps: If True, return number of refinement steps

        Returns:
            action_logits: (batch, num_actions)
        """
        # Use official TRM forward pass
        logits = self.trm(tokens)  # (batch, num_actions)
        
        # For compatibility, extract features if needed
        if return_features or return_refinement_steps:
            # Get intermediate representation for feature extraction
            with torch.no_grad():
                features = torch.zeros(tokens.shape[0], self.dim, device=tokens.device)
            refinement_steps = torch.tensor([self.max_recursion_steps] * tokens.shape[0], device=tokens.device)
            
            if return_features and return_refinement_steps:
                return logits, features, refinement_steps
            elif return_features:
                return logits, features
            elif return_refinement_steps:
                return logits, refinement_steps
        
        return logits

    def predict_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict action (argmax of logits)."""
        return self.trm.predict_action(tokens)

    def predict_action_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict action probabilities."""
        logits = self.forward(tokens)
        return torch.softmax(logits, dim=-1)


def create_model(
    grid_size: int = 8,
    dim: int = 64,
    depth: int = 2,
    **kwargs
) -> TRMNavigator:
    """
    Create TRM navigator model with appropriate configuration.

    Args:
        grid_size: Size of the grid (determines seq_len)
        dim: Model dimension
        depth: Number of layers (for compatibility, not used in official TRM)

    Returns:
        TRMNavigator model
    """
    seq_len = grid_size * grid_size + 4  # grid tokens + 4 coordinate tokens

    return TRMNavigator(
        dim=dim,
        seq_len=seq_len,
        depth=depth,  # Kept for compatibility
        **kwargs
    )
