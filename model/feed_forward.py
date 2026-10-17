"""TrustLM Feed-Forward Network.

Implements the position-wise feed-forward sublayer with GELU non-linearity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP) for Transformer blocks."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Standard convention: intermediate hidden dimension is 4x embedding dimension
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
