# TrustLM: A Custom Large Language Model for Reliable AI Responses

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)](#)

> **Important Architecture Notice**:
> **TrustLM is not a wrapper around any external pretrained LLM.**
> We are constructing a compact, decoder-only Transformer language model from the ground up using **pure PyTorch**. Every layer—from multi-head causal attention and feed-forward networks to tokenization, verification, memory, and unlearning—is natively implemented and fully inspectable.

---

## 🔬 Project Overview & Research Goals

Modern large language models struggle with unquantified hallucinations, black-box confidence estimates, and stubborn data contamination. When an LLM produces erroneous or outdated facts, retraining from scratch is computationally prohibitive, and fine-tuning often fails to guarantee that obsolete or sensitive data has actually been eliminated.

**TrustLM** investigates an integrated architecture designed from first principles for reliability, verifiable generation, explicit trust calibration, and targeted machine unlearning.

### Primary Research Goals

1. **Integrated Response Verification**
   - Incorporate predictive uncertainty estimation directly from decoder token logits and predictive entropy.
   - Perform multi-sample self-consistency checks and evidence verification to detect unsubstantiated claims and hallucinations *during* generation.

2. **Trust / Reliability Scoring**
   - Provide a calibrated, interpretable **Trust Score** $\in [0, 1]$ alongside every generated response.
   - Synthesize token confidence, sample agreement, and internal verification ground truth with Platt-scaled calibration.

3. **Selective Machine Unlearning**
   - Enable targeted fact and sequence erasure without requiring full model retraining.
   - Formulate gradient ascent objectives on designated *Forget Sets* while simultaneously bounding degradation on *Retain Sets* and preserving foundational linguistic competence via KL-divergence regularization.

### Supporting Architectural Component

4. **Controlled Modular Memory**
   - Decouple dynamic, editable knowledge from static model weights using an addressable key-value memory matrix.
   - Provides explicit read/write pathways, namespace isolation (e.g., session, factual, user), and clean erasure vectors to facilitate verifiable unlearning.

---

## 🔄 End-to-End Pipeline Architecture

```text
  [ Raw Data ]
       │
       ▼
 [ Tokenizer (tiktoken / BPE) ]
       │
       ▼
 [ Training Dataset (Train / Val / Retain / Forget) ]
       │
       ▼
 [ Custom Decoder-Only Transformer (PyTorch) ]
       │
       ▼
 [ Pretraining Loop (AdamW, Cosine LR, Grad Clip) ]
       │
       ▼
 [ Autoregressive Text Generation (Sampling / Top-k / Top-p) ]
       │
       ▼
 [ Integrated Response Verification (Entropy & Consistency) ]
       │
       ▼
 [ Trust & Reliability Scoring Engine ] ◄───► [ Controlled Modular Memory ]
       │                                                    ▲
       ▼                                                    │
 [ Selective Machine Unlearning (GA + Retain Loss + KL) ] ──┘
       │
       ▼
 [ Comprehensive Evaluation & Calibration Reports ]
```

---

## 📁 Repository Structure

```
TrustLM/
├── data/                  # Dataset management (raw corpora, tokenized tensors, split caches)
│   ├── README.md          # Guide to data layouts (raw, processed, retain/forget sets)
│   └── __init__.py
├── tokenizer/             # Byte-pair encoding wrappers and vocabulary utilities
│   └── __init__.py
├── model/                 # Custom decoder-only Transformer built from scratch (PyTorch)
│   └── __init__.py
├── training/              # Pretraining, optimizer configuration, LR scheduling, checkpointing
│   └── __init__.py
├── generation/            # Autoregressive generation, sampling, temperature, KV caching
│   └── __init__.py
├── verification/          # Research Goal 1: Response verification, entropy, consistency
│   └── __init__.py
├── trust/                 # Research Goal 2: Trust & reliability scoring, Platt calibration
│   └── __init__.py
├── memory/                # Supporting Component: Controlled modular key-value memory
│   └── __init__.py
├── unlearning/            # Research Goal 3: Selective machine unlearning (GA + Retain)
│   └── __init__.py
├── evaluation/            # Metrics (Perplexity, ECE, Forget Score, Retain Preservation)
│   └── __init__.py
├── utils/                 # Reproducibility helpers (seeds), device resolution, logging
│   └── __init__.py
├── tests/                 # Unit and integration test suites
│   ├── __init__.py
│   └── test_environment.py
├── config.py              # Centralized dataclass-based configuration system
├── requirements.txt       # Core project dependencies
├── README.md              # Research documentation and architecture guide
└── main.py                # Environment verification & foundation health check entrypoint
```

---

## ⚙️ Configuration System (`config.py`)

TrustLM adopts a strongly-typed, modular dataclass configuration hierarchy:

| Config Class | Scope | Key Parameters |
| :--- | :--- | :--- |
| `ModelConfig` | Transformer architecture | `vocab_size` (50257), `context_length` (256), `d_model` (256), `n_layers` (6), `n_heads` (8) |
| `TokenizerConfig` | Tokenizer backend | `backend` ("tiktoken"), `encoding_name` ("gpt2"), special tokens |
| `TrainingConfig` | Optimization & training | `batch_size` (16), `learning_rate` (3e-4), `max_iters` (5000), `device` ("cuda"/"cpu") |
| `GenerationConfig` | Decoding strategy | `max_new_tokens` (128), `temperature` (0.7), `top_k` (40), `top_p` (0.9) |
| `VerificationConfig` | Response verification | `entropy_threshold` (2.0), `token_confidence_threshold` (0.60), `logical_consistency_check` |
| `TrustConfig` | Trust score computation | `confidence_weight` (0.35), `consistency_weight` (0.35), `verification_weight` (0.30) |
| `MemoryConfig` | Modular memory store | `memory_dim` (256), `num_slots` (512), `read_heads` (2), `decay_rate` (0.01) |
| `UnlearningConfig` | Machine unlearning | `unlearn_lr` (1e-4), `forget_loss_weight` (1.0), `retain_loss_weight` (1.0), `kl_loss_weight` (0.5) |
| `PathConfig` | Storage directories | `data_dir`, `checkpoints_dir`, `logs_dir`, `plots_dir` |

Configurations can be serialized and exported to JSON or loaded programmatically:

```python
from config import TrustLMConfig, get_default_config

cfg = get_default_config()
cfg.save_json("config.json")
```

---

## 🚀 Environment Verification

To verify your environment, dependencies, and configuration health:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the environment verification runner
python main.py
```

`main.py` executes automated sanity checks covering:
- Python runtime (>= 3.10)
- PyTorch installation, device detection (CUDA / CPU), and tensor compute
- Core dependency presence (`tiktoken`, `numpy`, `tqdm`, `seaborn`, `matplotlib`, `pandas`)
- Repository package structure integrity
- Configuration instantiation, validation, and JSON serialization


---

## 🗺️ Implementation Roadmap

- [x] **Phase 1: Foundation Setup** *(Current)*
  - Modular project layout, centralized configuration, dependency pinning, and environment verification.
- [ ] **Phase 2: Tokenization & Custom Dataset Pipeline**
  - tiktoken BPE wrapper, data chunking, binary caching, retain/forget partition builders.
- [ ] **Phase 3: Custom Decoder-Only Transformer Architecture**
  - Pure PyTorch causal multi-head attention, rotary/learned embeddings, LayerNorm/RMSNorm, linear projections.
- [ ] **Phase 4: Pretraining Loop & Text Generation**
  - Autoregressive pretraining, AdamW, cosine scheduler, greedy/sampling/nucleus text generation with KV cache.
- [ ] **Phase 5: Integrated Response Verification & Trust Scoring**
  - Logit entropy computation, consistency checks across generation beams, Platt-calibrated trust metrics.
- [ ] **Phase 6: Controlled Modular Memory & Selective Unlearning**
  - Key-value external memory matrix, gradient ascent forget loss, retain preservation, catastrophic drift prevention.
- [ ] **Phase 7: Comprehensive Evaluation & Benchmarks**
  - Perplexity, calibration error (ECE), forget score, retain accuracy curves, and research report generation.
