"""Local TrustLM generator for the RAG pipeline."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

NO_TRAINED_CHECKPOINT_MESSAGE = (
    "No trained local TrustLM generation checkpoint is available."
)
MIN_GENERATION_TRAINING_STEPS = 1_000


class BaseGenerator(ABC):
    """Abstract generator interface for the RAG pipeline."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response string from a prompt."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this generator."""
        ...


class TrustLMGenerator(BaseGenerator):
    """Generate with a sufficiently trained local TrustLM checkpoint.

    Checkpoints with fewer than 1,000 recorded training steps are treated as
    smoke-test artifacts and are never used to produce a user-facing answer.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        device: Optional[str] = None,
    ) -> None:
        import torch

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k_sampling = top_k
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        self._tokenizer = None
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else self._find_checkpoint()
        )
        self._load_attempted = False

    @staticmethod
    def _find_checkpoint() -> Optional[Path]:
        """Find the latest non-smoke-test checkpoint in standard locations."""
        candidates = []
        for directory in (
            _PROJECT_ROOT / "checkpoints",
            _PROJECT_ROOT / "training" / "checkpoints",
        ):
            if directory.is_dir():
                candidates.extend(
                    path
                    for pattern in ("*.pt", "*.pth")
                    for path in directory.glob(pattern)
                    if "smoke_test" not in path.parts
                )
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

    @property
    def name(self) -> str:
        return "TrustLMGenerator (local)"

    @property
    def is_available(self) -> bool:
        """Whether a usable local checkpoint has been loaded."""
        return self._model is not None

    def _load_model(self) -> None:
        """Load a sufficiently trained TrustLM checkpoint, without random fallback."""
        if self._load_attempted:
            return

        if self._checkpoint_path is None:
            self._load_attempted = True
            logger.warning(NO_TRAINED_CHECKPOINT_MESSAGE)
            return
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"TrustLM checkpoint file not found: {self._checkpoint_path}"
            )

        import torch
        from model.trustlm import TrustLM
        from tokenizer.tokenizer import TrustLMTokenizer

        checkpoint = torch.load(
            self._checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        global_step = checkpoint.get("global_step")
        if not isinstance(global_step, int) or global_step < MIN_GENERATION_TRAINING_STEPS:
            self._load_attempted = True
            logger.warning(
                "%s Checkpoint %s has only %s training steps.",
                NO_TRAINED_CHECKPOINT_MESSAGE,
                self._checkpoint_path,
                global_step if global_step is not None else "unknown",
            )
            return
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"TrustLM checkpoint has no model_state_dict: {self._checkpoint_path}"
            )

        model_config = checkpoint.get("model_config", {})
        self._model = TrustLM(
            vocab_size=model_config.get("vocab_size", 50257),
            context_length=model_config.get("context_length", 256),
            embedding_dim=model_config.get("embedding_dim", 256),
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 6),
            dropout=0.0,
            bias=model_config.get("bias", False),
            tie_weights=model_config.get("tie_weights", True),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._model.to(self.device)
        self._tokenizer = TrustLMTokenizer()
        self._load_attempted = True
        logger.info(
            "Loaded local TrustLM checkpoint %s at step %d.",
            self._checkpoint_path,
            global_step,
        )

    def _synthesize_grounded_answer(
        self,
        prompt: str,
        chunks: list,
        question: str,
        context: str = "",
    ) -> str:
        """Synthesize a factual, grounded answer directly from retrieved document evidence."""
        if not chunks:
            return "The provided documents do not contain sufficient information to answer this question."

        # Filter or rank top relevant chunks
        top_chunks = chunks[:3]
        paragraphs = []
        for c in top_chunks:
            # Take clean text segments from chunk
            text = c.text.strip()
            # Clean excessive newlines
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            cleaned_body = " ".join(lines)
            if len(cleaned_body) > 400:
                cleaned_body = cleaned_body[:400].rsplit(".", 1)[0] + "."

            citation = f"({c.document_name}"
            if getattr(c, "page_number", None):
                citation += f", Page {c.page_number}"
            citation += ")"

            paragraphs.append(f"{cleaned_body} {citation}")

        return "\n\n".join(paragraphs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate from the local model or synthesize from retrieved document evidence."""
        import torch
        import torch.nn.functional as F

        chunks = kwargs.get("chunks", [])
        question = kwargs.get("question", "")
        context = kwargs.get("context", "")

        self._load_model()
        if self._model is None or self._tokenizer is None:
            # Fallback to grounded extractive synthesis from retrieved document evidence
            return self._synthesize_grounded_answer(prompt, chunks, question, context)

        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_k = kwargs.get("top_k", self.top_k_sampling)

        context_length = self._model.context_length
        token_ids = self._tokenizer.encode(prompt)
        if len(token_ids) > context_length - 1:
            token_ids = token_ids[-(context_length - 1):]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated_ids = token_ids[:]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = input_tensor[:, -context_length:]
                logits = self._model(idx_cond)[:, -1, :]

                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        top_k_clamped = min(top_k, logits.size(-1))
                        topk_vals, _ = torch.topk(logits, top_k_clamped)
                        threshold = topk_vals[:, -1].unsqueeze(-1)
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                next_id = next_token.item()
                if next_id == self._tokenizer.eos_token_id:
                    break

                generated_ids.append(next_id)
                input_tensor = torch.cat((input_tensor, next_token), dim=1)

        generated_text = self._tokenizer.decode(generated_ids[len(token_ids):]).strip()
        if not generated_text:
            return self._synthesize_grounded_answer(prompt, chunks, question, context)
        return generated_text
