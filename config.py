"""TrustLM: A Custom Large Language Model for Reliable AI Responses.

Central Configuration System.
Defines modular, strongly-typed hyperparameters and settings for:
  - Custom Transformer Architecture (compact decoder-only, built from scratch)
  - Tokenization
  - Pretraining and Optimization
  - Autoregressive Text Generation
  - Research Goal 1: Integrated Response Verification
  - Research Goal 2: Trust / Reliability Scoring
  - Research Goal 3: Selective Machine Unlearning
  - Supporting Component: Controlled Modular Memory
  - Evaluation and Visualization
  - Directory and Storage Paths
  - RAG (Retrieval-Augmented Generation) Pipeline
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ModelConfig:
    """Hyperparameters for the custom compact decoder-only Transformer."""

    vocab_size: int = 50257  # Byte-pair encoding vocabulary size
    context_length: int = 256  # Sequence context window (block size)
    d_model: int = 256  # Latent representation / embedding dimension
    n_layers: int = 6  # Number of Transformer decoder layers
    n_heads: int = 8  # Number of multi-head attention heads (head_dim = 32)
    d_ff: int = 1024  # Feedforward intermediate dimension (4 * d_model)
    dropout: float = 0.1  # Dropout regularization probability
    bias: bool = False  # Enable/disable bias in Linear and LayerNorm layers
    tie_word_embeddings: bool = True  # Tie input embedding and output projection weights


@dataclass
class TokenizerConfig:
    """Tokenizer configuration parameters."""

    backend: str = "tiktoken"  # Tokenizer backend ('tiktoken' or custom BPE)
    encoding_name: str = "gpt2"  # tiktoken base encoding scheme
    pad_token: str = "<|pad|>"
    bos_token: str = "<|bos|>"
    eos_token: str = "<|endoftext|>"
    unk_token: str = "<|unk|>"
    sep_token: str = "<|sep|>"


@dataclass
class TrainingConfig:
    """Pretraining and optimization loop configuration."""

    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_iters: int = 5000
    warmup_iters: int = 200
    lr_decay_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 50
    seed: int = 42
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerationConfig:
    """Autoregressive text generation decoding parameters."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class VerificationConfig:
    """Research Goal 1: Integrated Response Verification Configuration."""

    enable_verification: bool = True
    entropy_threshold: float = 2.0  # High token entropy flags model uncertainty
    token_confidence_threshold: float = 0.60  # Softmax threshold for fact-carrying tokens
    logical_consistency_check: bool = True  # Self-consistency comparison across samples
    fact_checking_enabled: bool = True  # Evidence ground-truth / memory lookup check
    hallucination_penalty: float = 0.35  # Penalty applied when unverified claims detected


@dataclass
class TrustConfig:
    """Research Goal 2: Trust / Reliability Scoring Configuration."""

    confidence_weight: float = 0.35  # Weight for model certainty (entropy / logits)
    consistency_weight: float = 0.35  # Weight for sample consistency and agreement
    verification_weight: float = 0.30  # Weight for verified statements vs ground-truth
    min_acceptable_trust_score: float = 0.65  # Score threshold below which responses are flagged
    flag_low_trust: bool = True  # Mark unreliable responses explicitly
    calibration_method: str = "platt_scaling"  # Calibration technique for confidence scores


@dataclass
class MemoryConfig:
    """Supporting Component: Controlled Modular Memory Configuration."""

    enabled: bool = True
    memory_type: str = "modular_key_value"  # Controlled read/write key-value addressable memory
    memory_dim: int = 256  # Dimension of memory slots (matches d_model)
    num_slots: int = 512  # Maximum capacity of memory module
    read_heads: int = 2  # Attention read heads accessing memory
    write_heads: int = 1  # Gated write heads updating memory
    decay_rate: float = 0.01  # Memory decay rate for temporary knowledge
    isolation_namespaces: List[str] = field(
        default_factory=lambda: ["user", "factual", "session"]
    )


@dataclass
class UnlearningConfig:
    """Research Goal 3: Selective Machine Unlearning Configuration."""

    method: str = "gradient_ascent_retain"  # GA on forget set + CE on retain set + KL divergence
    unlearn_lr: float = 1e-4
    forget_loss_weight: float = 1.0  # Maximizes loss on the forget dataset
    retain_loss_weight: float = 1.0  # Preserves utility on retain dataset
    kl_loss_weight: float = 0.5  # Prevents catastrophic divergence from original model
    unlearn_steps: int = 100
    target_forget_accuracy: float = 0.05  # Chance level target on forgotten samples
    max_retain_degradation: float = 0.03  # Tolerable performance drop on retained knowledge


@dataclass
class EvaluationConfig:
    """Comprehensive evaluation metrics and visualization configuration."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "perplexity",
            "cross_entropy_loss",
            "trust_score",
            "verification_rate",
            "unlearning_efficacy",
            "retain_preservation",
        ]
    )
    save_visualizations: bool = True
    plot_dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in getattr(torch, "__doc__", "") else "default"


@dataclass
class RAGConfig:
    """Configuration for the Retrieval-Augmented Generation (RAG) pipeline.

    These settings control document ingestion, chunk splitting, semantic
    retrieval embeddings, and the ChromaDB vector store.

    NOTE: The 'embedding_model' here refers to the sentence-transformers model
    used for SEMANTIC RETRIEVAL — it is completely separate from the
    TokenEmbedding / PositionalEmbedding layers inside the TrustLM Transformer.
    """

    # Semantic retrieval embedding model (separate from TrustLM token embeddings)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking parameters
    # chunk_size: target characters per chunk (approx. 200–300 tokens for MiniLM)
    chunk_size: int = 1000
    # chunk_overlap: characters re-included from previous chunk to avoid boundary cuts
    chunk_overlap: int = 200

    # Retrieval parameters
    top_k: int = 5  # Number of chunks to retrieve for each query

    # ChromaDB vector store settings
    vector_db_path: str = "data/knowledge_base/chroma_db"  # Relative to project root
    collection_name: str = "trustlm_knowledge_base"

    # Context builder
    max_context_chars: int = 8000  # Hard cap on total context characters per query


@dataclass
class PathConfig:
    """Filesystem directory paths relative to project root."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    checkpoints_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "training" / "checkpoints"
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "evaluation" / "logs"
    )
    plots_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "evaluation" / "plots"
    )
    knowledge_base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "knowledge_base"
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not already exist."""
        for p in [
            self.data_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir,
            self.knowledge_base_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class TrustLMConfig:
    """Master configuration class aggregating all modular sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dictionary."""
        data = asdict(self)
        # Convert Path objects to strings for JSON serializability
        if "paths" in data:
            data["paths"] = {k: str(v) for k, v in data["paths"].items()}
        return data

    def save_json(self, filepath: Path | str) -> None:
        """Export configuration to JSON file."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrustLMConfig:
        """Construct configuration from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            verification=VerificationConfig(**data.get("verification", {})),
            trust=TrustConfig(**data.get("trust", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            unlearning=UnlearningConfig(**data.get("unlearning", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            rag=RAGConfig(**data.get("rag", {})),
        )

    def validate(self) -> bool:
        """Perform sanity checks on hyperparameters."""
        assert self.model.d_model % self.model.n_heads == 0, (
            f"d_model ({self.model.d_model}) must be divisible by n_heads ({self.model.n_heads})"
        )
        assert self.model.context_length > 0, "context_length must be positive"
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"
        assert 0.0 <= self.generation.temperature <= 2.0, "temperature must be between 0 and 2.0"
        assert abs(
            (self.trust.confidence_weight + self.trust.consistency_weight + self.trust.verification_weight) - 1.0
        ) < 1e-4, "Trust weights must sum to 1.0"
        return True


def get_default_config() -> TrustLMConfig:
    """Return an initialized and validated default configuration."""
    cfg = TrustLMConfig()
    cfg.validate()
    return cfg
