"""TrustLM Transformer Block.

Implements a single decoder Transformer block featuring Pre-LayerNorm,
causal multi-head self-attention, feed-forward sublayer, and residual connections.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """A single decoder-only Transformer block with Pre-LayerNorm architecture.

    Order of operations:
      1. LayerNorm (ln_1)
      2. Multi-Head Self-Attention (attn)
      3. Residual connection: x = x + attn(ln_1(x))
      4. LayerNorm (ln_2)
      5. Feed-Forward Network (mlp)
      6. Residual connection: x = x + mlp(ln_2(x))
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # 1. First LayerNorm
        self.ln_1 = nn.LayerNorm(embedding_dim, bias=bias)

        # 2. Causal Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            bias=bias,
        )

        # 4. Second LayerNorm
        self.ln_2 = nn.LayerNorm(embedding_dim, bias=bias)

        # 5. Position-wise Feed-Forward Network
        self.mlp = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a Transformer block.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            return_attention_weights: Whether to return computed attention weights.

        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # 1-3. Pre-LN -> Self-Attention -> Residual Connection
        if return_attention_weights:
            attn_out, weights = self.attn(self.ln_1(x), return_attention_weights=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln_1(x))

        # 4-6. Pre-LN -> Feed-Forward -> Residual Connection
        x = x + self.mlp(self.ln_2(x))

        if return_attention_weights:
            return x, weights

        return x
