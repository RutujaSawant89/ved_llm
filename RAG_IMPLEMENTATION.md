# TrustLM RAG Implementation

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances a language model by giving it access to an external knowledge base at inference time. Instead of relying solely on knowledge baked into model weights, a RAG system:
1. Retrieves relevant documents from a knowledge base using semantic search.
2. Provides those documents as context to the generator.
3. Instructs the generator to answer using only the retrieved evidence.

## 2. Architecture

`
USER QUERY
    |
QUERY EMBEDDING          (all-MiniLM-L6-v2)
    |
VECTOR DATABASE SEARCH   (ChromaDB - cosine similarity)
    |
TOP-K RELEVANT CHUNKS
    |
CONTEXT BUILDER          (formats chunks with source metadata)
    |
PROMPT CONSTRUCTION      (system instruction + context + question)
    |
GENERATION MODEL         (TrustLMGenerator)
    |
FINAL RESPONSE + SOURCE EVIDENCE
`

## 3. Two Types of Embeddings

CRITICAL DISTINCTION:
- Token Embeddings (model/embeddings.py): Map discrete token IDs inside the Transformer for next-token prediction
- Retrieval Embeddings (rag/embeddings.py): Map text to semantic vectors for ChromaDB similarity search

These are completely separate systems.

## 4. Current Limitation

The only available checkpoint is a smoke-test artifact trained for 4 steps. It is not used for generation because it is not sufficiently trained to produce meaningful answers. Until a trained local checkpoint is available, generation reports:

`No trained local TrustLM generation checkpoint is available.`

Retrieval remains fully functional and does not require a generation checkpoint.

## 5. Commands

Run demo:     cd TrustLM && python -m rag.demo
Run tests:    cd TrustLM && pytest -q
Smoke test:   cd TrustLM && python training/train.py --smoke-test
