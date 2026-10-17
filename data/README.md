# Data Directory

This directory stores datasets used across the TrustLM lifecycle:

- `raw/`: Raw text corpora used for pretraining.
- `processed/`: Memory-mapped or pre-tokenized binary/tensor batches (`train.bin`, `val.bin`).
- `verification/`: Benchmark datasets containing known factual statements, counterfactual claims, and consistency evaluation pairs.
- `unlearning/`:
  - `forget_set.jsonl`: Target knowledge snippets to selectively erase.
  - `retain_set.jsonl`: Core knowledge to strictly preserve without degradation.
