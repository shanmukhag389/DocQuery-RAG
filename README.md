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

                       [ DOCUMENT INGESTION ]
                                 │
                                 ▼
         PDF ──> PyMuPDF ──> LangChain (1000 chars / 150 overlap)
                                 │
                                 ▼
             Gemini Embeddings (gemini-embedding-001, 768-dim)
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
             Qdrant Cloud                 BM25 Index
         (Dense/Vector Search)       (Sparse/Keyword Search)
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                    Reciprocal Rank Fusion (RRF)
                                 │
                                 ▼
                FlashRank Cross-Encoder Re-ranking
                                 │
                                 ▼
                    Gemini Generation (Grounded)
                                 │
                                 ▼
               LLM-as-Judge (Faithfulness + Relevance)


## Tech Stack

| Component | Tool |
|---|---|
| LLM & Embeddings | Google Gemini API (`gemini-3.6-flash`, `gemini-embedding-001`) |
| Vector Database | Qdrant Cloud (free tier) |
| Sparse Search | BM25 (`rank-bm25`) |
| Re-ranking | FlashRank (`ms-marco-MiniLM-L-12-v2` cross-encoder) |
| Document Processing | PyMuPDF, LangChain Text Splitters |
| Environment | Google Colab |

## Results

Tested against a real 38-page research paper ([APS-RAG](https://arxiv.org/abs/2607.24663), Argonne National Laboratory), including a deliberately out-of-scope question to test hallucination resistance:

| Query | Faithfulness | Context Relevance | Notes |
|---|---|---|---|
| "What is APS-Bench and how many questions does it contain?" | 1.0 | 1.0 | Correctly extracted exact figure: 50 questions (49 answerable + 1 abstention) |
| "What cross-encoder reranker model is used in this system?" | 1.0 | 1.0 | Correctly identified specific model name (jina-reranker-v3) among 3 candidates discussed in the paper |
| "How does removing the reranker affect vital recall?" | 1.0 | 1.0 | Correctly extracted exact statistic: 32.8% recall drop, with confidence interval |
| "What is the capital of France?" (out-of-scope) | 1.0 | 0.0 | Correctly refused to answer rather than hallucinating from general knowledge |

### Hybrid search vs. dense-only: a concrete example

Querying `"APS-Bench"` against **dense (vector) search alone** returned two off-topic chunks (about data loggers, bunch current measurements) scoring nearly as high (0.75, 0.746) as genuinely relevant chunks (0.771) — the classic vector-search weakness of loosely-similar noise crowding out exact matches.

After **hybrid RRF fusion + cross-encoder re-ranking**, the re-ranker produced sharply decisive scores (0.997 → 0.024), clearly separating truly relevant chunks from noise — something RRF's rank-based scores alone (0.0164 vs 0.0159) could not do.

## How to run

1. Open `DocQuery_RAG_Notebook.ipynb` in Google Colab (badge at the top of the notebook)
2. Add three secrets in Colab's Secrets panel (🔑 icon): `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
3. Run cells top to bottom
4. Upload any PDF to `/content/` and update `pdf_path` to test with your own document

## Roadmap

- [ ] Wrap pipeline in FastAPI (`/upload`, `/query`, `/health`, `/evaluate`)
- [ ] Containerize with Docker
- [ ] Deploy to AWS (Lambda + S3 + API Gateway)
