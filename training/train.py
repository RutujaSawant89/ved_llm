"""TrustLM Pretraining Entrypoint.

CLI runner supporting:
  - Automated device selection (CUDA / CPU)
  - Data safety verification before training initiation
  - Smoke-test mode for fast CPU pipeline verification
  - Full research-scale pretraining
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import get_default_config
from model.trustlm import TrustLM
from tokenizer.tokenizer import TrustLMTokenizer
from training.checkpoint import CheckpointManager
from training.device import get_device_info, get_training_device, print_device_summary
from training.trainer import AutoregressiveDataset, TrustLMTrainer, verify_dataset_safety


def run_smoke_test(device: torch.device) -> int:
    """Execute a lightweight CPU-compatible smoke test to verify training loop integrity."""
    print("=" * 70)
    print("           TRUSTLM PRETRAINING: RUNNING SMOKE TEST")
    print("=" * 70)
    print("Mode: Smoke Test (5 verification steps on CPU/CUDA)")

    tokenizer = TrustLMTokenizer()
    data_dir = project_root / "data"
    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"

    # 1. Verify data safety
    train_tokens = verify_dataset_safety(
        train_file, tokenizer, vocab_size=50257, context_length=64
    )
    val_tokens = verify_dataset_safety(
        val_file, tokenizer, vocab_size=50257, context_length=64
    )
    print(f"Data Safety Verified: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens.")

    # 2. Build smoke test datasets with compact context length
    train_ds = AutoregressiveDataset(
        token_ids=train_tokens,
        context_length=64,
        vocab_size=50257,
    )
    val_ds = AutoregressiveDataset(
        token_ids=val_tokens,
        context_length=64,
        vocab_size=50257,
    )

    # 3. Instantiate compact smoke model
    model = TrustLM(
        vocab_size=50257,
        context_length=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        tie_weights=True,
    )
    print(f"Smoke Test Model: {model.num_parameters:,} trainable parameters.")

    # 4. Run 5-step trainer
    trainer = TrustLMTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=1e-3,
        batch_size=2,
        max_iters=5,
        eval_interval=2,
        eval_iters=2,
        log_interval=1,
        device=device,
        checkpoint_dir=project_root / "checkpoints" / "smoke_test",
        plots_dir=project_root / "evaluation" / "plots",
    )

    results = trainer.train(max_steps=5, num_epochs=1)

    print("\n" + "=" * 70)
    print("                   SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f" Steps Completed     : {results['global_step']}")
    print(f" Tokens Processed    : {results['tokens_trained']:,}")
    print(f" Final Train Loss    : {results['final_train_loss']:.4f}")
    print(f" Final Val Loss      : {results['final_val_loss']:.4f}")
    print(f" Final Perplexity    : {results['final_perplexity']:.2f}")
    print("=" * 70 + "\n")
    return 0


def run_full_training(
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 1,
    resume: bool = False,
    device_override: str | None = None,
) -> int:
    """Execute the full research pretraining pipeline."""
    cfg = get_default_config()
    device = get_training_device(device_override)

    print("=" * 70)
    print("       TRUSTLM RESEARCH PRETRAINING PIPELINE (FULL SCALE)")
    print("=" * 70)
    print_device_summary(get_device_info(device_override))

    tokenizer = TrustLMTokenizer()
    data_dir = project_root / "data"
    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"

    # Verify data safety
    print("Verifying data safety and token integrity...")
    train_tokens = verify_dataset_safety(
        train_file, tokenizer, vocab_size=cfg.model.vocab_size, context_length=cfg.model.context_length
    )
    val_tokens = verify_dataset_safety(
        val_file, tokenizer, vocab_size=cfg.model.vocab_size, context_length=cfg.model.context_length
    )
    print(f"Data Safety Verified: {len(train_tokens):,} train tokens | {len(val_tokens):,} val tokens.")

    # Datasets
    train_ds = AutoregressiveDataset(
        token_ids=train_tokens,
        context_length=cfg.model.context_length,
        vocab_size=cfg.model.vocab_size,
    )
    val_ds = AutoregressiveDataset(
        token_ids=val_tokens,
        context_length=cfg.model.context_length,
        vocab_size=cfg.model.vocab_size,
    )

    # Instantiate full research model
    model = TrustLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        embedding_dim=cfg.model.d_model,
        num_heads=cfg.model.n_heads,
        num_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
        tie_weights=cfg.model.tie_word_embeddings,
    )
    model.print_architecture()

    trainer = TrustLMTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_iters=cfg.training.max_iters,
        warmup_iters=cfg.training.warmup_iters,
        eval_interval=cfg.training.eval_interval,
        eval_iters=cfg.training.eval_iters,
        log_interval=cfg.training.log_interval,
        device=device,
        checkpoint_dir=project_root / "checkpoints",
        plots_dir=project_root / "evaluation" / "plots",
    )

    if resume:
        latest = trainer.checkpoint_manager.get_latest_checkpoint()
        if latest:
            print(f"Resuming training from checkpoint: {latest}")
            trainer.checkpoint_manager.load_checkpoint(
                latest, model=model, optimizer=trainer.optimizer, device=device
            )

    results = trainer.train(num_epochs=epochs)
    print("Full training completed successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustLM Pretraining CLI")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 5-step CPU/CUDA smoke test to verify training loop without full training",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--device", type=str, default=None, help="Force specific device ('cpu', 'cuda', etc.)"
    )

    args = parser.parse_args()
    device = get_training_device(args.device)

    if args.smoke_test:
        return run_smoke_test(device)

    return run_full_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        resume=args.resume,
        device_override=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
