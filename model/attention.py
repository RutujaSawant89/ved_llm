"""TrustLM Multi-Head Causal Self-Attention.

Implements multi-head self-attention with lower-triangular causal masking,
ensuring tokens can only attend to preceding and current positions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention layer."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # Combined query, key, value projection
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        # Output projection back to embedding_dim
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of ones
        # Upper triangle is zero, ensuring tokens cannot attend to future positions
        causal_mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, context_length, context_length))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            return_attention_weights: Whether to return computed attention probabilities.

        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
            Optionally returns (output, attention_weights).
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max context length {self.context_length}"
            )

        # 1. Project input to Q, K, V: shape (batch_size, seq_len, 3 * embedding_dim)
        qkv = self.qkv_proj(x)

        # 2. Split into Q, K, V and reshape for multi-head attention:
        # (batch_size, seq_len, 3, num_heads, head_dim) -> permute to (3, batch_size, num_heads, seq_len, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 3. Scaled dot-product attention scores: (batch_size, num_heads, seq_len, seq_len)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 4. Apply causal masking: mask future tokens with -infinity before softmax
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax attention probabilities and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # 6. Compute context output: (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights_dropped, v)

        # 7. Concatenate heads and project back to embedding dimension:
        # (batch_size, seq_len, num_heads * head_dim) = (batch_size, seq_len, embedding_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        out = self.resid_dropout(self.out_proj(out))

        if return_attention_weights:
            return out, attn_weights

        return out
