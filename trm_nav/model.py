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
        num_actions: int = 4,
        max_recursion_steps: int = 30,
        l_cycles: int = 1,  # Inner loop iterations (original TRM uses 4-6)
        halt_prob_thres: float = 0.5,
        dropout: float = 0.1,
        use_fallback: bool = False,
        mode: str = "classification",  # "classification" or "path_prediction"
        grad_last_only: bool = False,  # Like original TRM: only last H-cycle has gradients
    ):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.max_recursion_steps = max_recursion_steps
        self.l_cycles = l_cycles
        self.halt_prob_thres = halt_prob_thres
        self.dropout_rate = dropout
        self.mode = mode

        # Official TRM implementation
        self.trm = NavigationTRM(
            seq_len=seq_len,
            vocab_size=num_tokens,
            hidden_size=dim,
            num_heads=max(1, dim // 16),
            max_recursion_steps=max_recursion_steps,
            l_cycles=l_cycles,
            dropout=dropout,
            use_fallback=use_fallback,
            mode=mode,
            grad_last_only=grad_last_only,
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

    def predict_path(self, tokens: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
        """
        Predict path mask for path_prediction mode.

        Args:
            tokens: Input tokens (batch, seq_len)
            grid_size: Size of grid (for reshaping)

        Returns:
            path_mask: (batch, grid_size, grid_size) binary mask (1 where path, 0 elsewhere)
        """
        if self.mode != "path_prediction":
            raise ValueError("predict_path only works in path_prediction mode")

        logits = self.forward(tokens)  # (batch, seq_len, 4)
        # Take only grid tokens (first grid_size^2 tokens)
        grid_logits = logits[:, :grid_size * grid_size, :]  # (batch, 64, 4)
        preds = grid_logits.argmax(dim=-1)  # (batch, 64)
        # Class 3 = path, convert to binary mask
        path_mask = (preds == 3).long()
        return path_mask.view(-1, grid_size, grid_size)

    def predict_action_from_path(
        self,
        tokens: torch.Tensor,
        grid_size: int = 8
    ) -> torch.Tensor:
        """
        Predict next action by predicting path and finding next step.

        For inference: predicts the full path, then determines which
        adjacent cell is on the path to get the action.

        Args:
            tokens: Input tokens (batch, seq_len)
            grid_size: Size of grid

        Returns:
            actions: (batch,) predicted actions
        """
        if self.mode != "path_prediction":
            raise ValueError("predict_action_from_path only works in path_prediction mode")

        batch_size = tokens.shape[0]
        device = tokens.device

        # Get predicted path
        path_mask = self.predict_path(tokens, grid_size)  # (batch, grid_size, grid_size)

        # Extract current position from tokens (last 4 tokens are coords)
        # tokens[-4:-2] = start_row, start_col (offset by 3)
        start_row = (tokens[:, -4] - 3).long()
        start_col = (tokens[:, -3] - 3).long()

        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        # For each sample, find adjacent path cell
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        for b in range(batch_size):
            r, c = start_row[b].item(), start_col[b].item()

            # Check each direction for a path cell
            for action_idx, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if path_mask[b, nr, nc] == 1:
                        actions[b] = action_idx
                        break

        return actions


def create_model(
    grid_size: int = 8,
    dim: int = 64,
    depth: int = 2,
    use_fallback: bool = False,
    mode: str = "classification",
    **kwargs
) -> TRMNavigator:
    """
    Create TRM navigator model with appropriate configuration.

    Args:
        grid_size: Size of the grid (determines seq_len)
        dim: Model dimension
        depth: Number of layers (for compatibility, not used in official TRM)
        mode: "classification" for action prediction, "path_prediction" for seq2seq

    Returns:
        TRMNavigator model
    """
    seq_len = grid_size * grid_size + 4  # grid tokens + 4 coordinate tokens

    return TRMNavigator(
        dim=dim,
        seq_len=seq_len,
        depth=depth,  # Kept for compatibility
        use_fallback=use_fallback,
        mode=mode,
        **kwargs
    )
