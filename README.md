# DocQuery-RAG

An advanced Retrieval-Augmented Generation (RAG) system that goes beyond naive vector search — combining **dense + sparse hybrid retrieval**, **cross-encoder re-ranking**, and an **LLM-as-judge evaluation framework** to answer questions over PDF documents with grounded, hallucination-resistant responses.

Built and validated end-to-end in Google Colab using the Gemini API and Qdrant Cloud.

## Why this isn't a naive PDF chatbot

Most tutorial-level RAG projects use pure vector (embedding) search, which has a well-known failure mode: it's bad at exact-match lookups — serial numbers, model names, specific terminology — because embeddings capture semantic meaning, not exact tokens. This project explicitly demonstrates and fixes that gap:

- **Hybrid Search (Dense + Sparse + RRF):** Combines Qdrant's vector search (semantic meaning) with BM25 keyword search (exact terms), fused using Reciprocal Rank Fusion — a rank-based combination method that works even though the two methods' raw scores are on completely different, incomparable scales.
- **Cross-Encoder Re-ranking (FlashRank):** RRF narrows candidates but still carries noise (see results below). A lightweight cross-encoder re-reads the query against each candidate chunk *together*, producing sharply more confident relevance scores and filtering out near-miss noise before it reaches the LLM.
- **Grounded Generation:** Gemini is instructed to answer *only* from retrieved context and explicitly refuse when the answer isn't present — verified with an out-of-scope query test (below).
- **LLM-as-Judge Evaluation:** Faithfulness and context-relevance are scored automatically per query, the same underlying approach used by evaluation frameworks like RAGAS.

## Architecture
