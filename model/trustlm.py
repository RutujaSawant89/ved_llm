"""TrustLM: Custom Decoder-Only Transformer Architecture.

Built completely from scratch in PyTorch without external pretrained wrappers.
Implements:
  Token Embedding + Positional Embedding
  ↓
  Transformer Block × N
  ↓
  Final LayerNorm
  ↓
  Linear Language Modeling Head
  ↓
  Vocabulary Logits of shape (batch_size, sequence_length, vocab_size)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import PositionalEmbedding, TokenEmbedding
from model.transformer_block import TransformerBlock


class TrustLM(nn.Module):
    """Custom decoder-only Transformer language model for TrustLM."""

    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 256,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        tie_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize TrustLM model with adjustable hyperparameters.

        Args:
            vocab_size: Size of the token vocabulary.
            context_length: Maximum context sequence length (block size).
            embedding_dim: Dimension of latent representations and embeddings (d_model).
            num_heads: Number of causal multi-head attention heads.
            num_layers: Number of stacked Transformer blocks.
            dropout: Dropout probability applied to embeddings and residual connections.
            hidden_dim: Intermediate dimension of feed-forward networks (defaults to 4 * embedding_dim).
            bias: Whether linear layers and layer norms include learnable bias.
            tie_weights: Whether to tie token embedding weights with the LM head projection.
        """
        super().__init__()

        # Handle alternative parameter aliases (e.g., from ModelConfig)
        if "d_model" in kwargs:
            embedding_dim = kwargs.pop("d_model")
        if "n_heads" in kwargs:
            num_heads = kwargs.pop("n_heads")
        if "n_layers" in kwargs:
            num_layers = kwargs.pop("n_layers")
        if "d_ff" in kwargs:
            hidden_dim = kwargs.pop("d_ff")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.tie_weights = tie_weights

        # 1. Token and Positional Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # 2. Stacked Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final Pre-Head LayerNorm
        self.ln_f = nn.LayerNorm(embedding_dim, bias=bias)

        # 4. Linear Language Modeling Head
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=bias)

        # Optional weight tying between token embedding and LM head
        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with standard normal distribution (std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Tensor of token indices with shape (batch_size, sequence_length).
            targets: Optional ground-truth token indices with shape (batch_size, sequence_length)
                     used to calculate cross-entropy loss.

        Returns:
            If targets is None:
                logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            If targets is provided:
                (logits, loss): Tuple containing vocabulary logits and scalar loss.
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; maximum context length is {self.context_length}"
            )

        # 1. Embeddings: Token embedding + Positional embedding
        tok_emb = self.token_embedding(input_ids)  # (B, T, C)
        pos_emb = self.positional_embedding(seq_len, device=input_ids.device)  # (1, T, C)
        x = self.emb_dropout(tok_emb + pos_emb)

        # 2. Sequential Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm
        x = self.ln_f(x)

        # 4. LM Head Projection to Vocabulary Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            # Flatten predictions and targets to compute cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculate the number of parameters in the model.

        Args:
            non_embedding: If True, excludes position and token embedding weights
                           to reflect core Transformer block compute capacity.

        Returns:
            Total count of trainable parameters.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.positional_embedding.embedding.weight.numel()
            n_params -= self.token_embedding.embedding.weight.numel()
        return n_params

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return self.get_num_params(non_embedding=False)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Return a structured parameter count summary by module."""
        token_emb_params = self.token_embedding.embedding.weight.numel()
        pos_emb_params = self.positional_embedding.embedding.weight.numel()

        block_params = sum(
            sum(p.numel() for p in block.parameters() if p.requires_grad)
            for block in self.blocks
        )
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters() if p.requires_grad)
        head_params = (
            0 if self.tie_weights else sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        )

        total_params = self.num_parameters
        non_emb_params = self.get_num_params(non_embedding=True)

        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "tie_weights": self.tie_weights,
            "token_embeddings": token_emb_params,
            "positional_embeddings": pos_emb_params,
            "transformer_blocks_total": block_params,
            "transformer_block_per_layer": block_params // max(1, self.num_layers),
            "final_layernorm": ln_f_params,
            "lm_head_unique": head_params,
            "total_parameters": total_params,
            "non_embedding_parameters": non_emb_params,
        }

    def print_architecture(self) -> None:
        """Print a formatted summary of the model architecture and parameter allocation."""
        summary = self.get_parameter_summary()
        line = "=" * 70
        print(f"\n{line}")
        print("          TrustLM Custom Transformer Architecture Summary")
        print(line)
        print(f" Vocabulary Size         : {summary['vocab_size']:,}")
        print(f" Context Length          : {summary['context_length']}")
        print(f" Embedding Dimension     : {summary['embedding_dim']}")
        print(f" Attention Heads         : {summary['num_heads']} (Head Dim: {self.embedding_dim // self.num_heads})")
        print(f" Transformer Layers      : {summary['num_layers']}")
        print(f" Weight Tying (Embed/LM) : {summary['tie_weights']}")
        print("-" * 70)
        print(f" Token Embedding Params  : {summary['token_embeddings']:,}")
        print(f" Positional Embed Params : {summary['positional_embeddings']:,}")
        print(f" Blocks Total Params     : {summary['transformer_blocks_total']:,} ({self.num_layers} layers)")
        print(f" Final LayerNorm Params  : {summary['final_layernorm']:,}")
        print("-" * 70)
        print(f" Total Trainable Params  : {summary['total_parameters']:,}")
        print(f" Non-Embedding Params    : {summary['non_embedding_parameters']:,}")
        print(f"{line}\n")
