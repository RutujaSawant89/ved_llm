"""TrustLM Embedding Layers.

Implements token embeddings and learnable positional embeddings
for the custom decoder-only Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps discrete token IDs to dense continuous representation vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            Tensor of token embeddings with shape (batch_size, sequence_length, embedding_dim).
        """
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """Maps sequence position indices to dense positional vectors."""

    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Forward pass.

        Args:
            sequence_length: Current sequence length (must be <= context_length).
            device: Compute device where indices should reside.

        Returns:
            Tensor of positional embeddings with shape (1, sequence_length, embedding_dim).
        """
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max context length {self.context_length}"
            )
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=device)
        return self.embedding(positions).unsqueeze(0)
