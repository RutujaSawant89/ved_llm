# TrustLM RAG Implementation Status

## Active RAG flow

The demo retains the existing `all-MiniLM-L6-v2` embeddings, ChromaDB
knowledge base, retriever, query processor, reranker, context builder, and
prompt builder. Generation is wired to the repository's local
`TrustLMGenerator` and custom PyTorch decoder-only Transformer.

## Local checkpoint status

The only checkpoint found is under `checkpoints/smoke_test` and records 4
training steps. The local generator does not use it or initialize random
weights for answering. A local checkpoint must record at least 1,000 training
steps before it can be used for generation. Until then, generation reports:

`No trained local TrustLM generation checkpoint is available.`

## Verification status

Claim verification currently reports that local verification is not
implemented. It makes no external service call. The RAG demo requires no API
key.

## Running

From the TrustLM project directory:

```bash
python -m pytest
python -m rag.demo
```
