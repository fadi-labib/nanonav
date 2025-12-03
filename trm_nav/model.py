"""
TRM Model Wrapper

Wraps the tiny-recursive-model library for navigation task.
Requires tiny-recursive-model to be installed - no fallback.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import tiny-recursive-model - fail hard if not available
try:
    from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D
except ImportError:
    raise ImportError(
        "tiny-recursive-model is required but not installed.\n"
        "Install it with: pip install tiny-recursive-model\n"
        "Or from source: pip install git+https://github.com/lucidrains/tiny-recursive-model.git"
    )


class TRMNavigator(nn.Module):
    """
    TRM-based navigation model.

    Takes encoded state (tokens) and predicts action logits.

    Uses recursive refinement from tiny-recursive-model to iteratively
    improve the representation before classification.
    """

    def __init__(
        self,
        dim: int = 64,
        num_tokens: int = 256,
        seq_len: int = 68,  # 8*8 + 4 for 8x8 grid
        depth: int = 2,
        num_actions: int = 5,
        max_recursion_steps: int = 8,
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

        # TRM with MLP-Mixer network
        self.trm = TinyRecursiveModel(
            dim=dim,
            num_tokens=num_tokens,
            network=MLPMixer1D(
                dim=dim,
                depth=depth,
                seq_len=seq_len
            )
        )
        self.feature_dropout = nn.Dropout(dropout)

        # Classification head: pool sequence then classify
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_actions)
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
        # Get embeddings
        embedded = self.trm.input_embed(tokens)  # (batch, seq_len, dim)

        # Apply the network (MLP-Mixer) for recursive refinement
        features = embedded
        for _ in range(self.max_recursion_steps):
            features = self.trm.network(features)

        # Pool over sequence dimension
        features = features.mean(dim=1)  # (batch, dim)
        features = self.feature_dropout(features)

        refinement_steps = torch.tensor([self.max_recursion_steps] * tokens.shape[0])

        # Classify
        logits = self.classifier(features)

        if return_features and return_refinement_steps:
            return logits, features, refinement_steps
        elif return_features:
            return logits, features
        elif return_refinement_steps:
            return logits, refinement_steps
        return logits

    def predict_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict action (argmax of logits)."""
        logits = self.forward(tokens)
        return logits.argmax(dim=-1)

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
        depth: Number of mixer layers

    Returns:
        TRMNavigator model
    """
    seq_len = grid_size * grid_size + 4  # grid tokens + 4 coordinate tokens

    return TRMNavigator(
        dim=dim,
        seq_len=seq_len,
        depth=depth,
        **kwargs
    )
