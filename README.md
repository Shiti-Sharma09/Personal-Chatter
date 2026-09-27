# Personal Chatter

A local, open-source RAG chatbot: upload your own documents in-chat and ask questions about them, grounded in citations. Runs entirely offline — no API keys, no data leaving your machine.

## What changed (naive RAG → advanced RAG)

The original version of this project was a straightforward 2023-era LangChain quickstart clone. This upgrade closes the gap between that and what's actually state-of-the-art in RAG today:

| | Before | Now |
|---|---|---|
| Retrieval | Dense-only FAISS lookup, hardcoded `k=2` | **Hybrid search**: dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion |
| Ranking | Whatever the embedding model returned, unranked further | **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the fused candidates |
| Chunking | Fixed-size splitting only | Fixed-size splitting **plus contextual chunk headers** (source, page, section) prepended before embedding — a cheap form of Anthropic's "contextual retrieval," shown to cut retrieval-miss rate |
| Conversation | Every message treated as a fresh, context-free query | **Conversational memory** — follow-ups are condensed into standalone questions using chat history |
| File upload | README claimed it; code only read a pre-populated `Data/` folder via an offline script | **Real in-chat upload** via Chainlit's `AskFileMessage`, plus PDF/TXT/MD/DOCX support (was PDF-only) |
| Citations | Raw file paths, naive dedup | `filename (page N)`, deduped |
| LLM runtime | `CTransformers` + GGML (deprecated ~2 years ago) | `llama-cpp-python` + GGUF |
| Streaming | Langchain callback paired with a backend that didn't really support token streaming | Real token streaming via a background thread into the Chainlit message |
| Tests | None | `pytest` suite covering ingestion, hybrid fusion, reranking, memory, and the full pipeline |

## Prerequisites

- Python 3.10+
- A local GGUF model (e.g. a Llama-3 or Mistral instruct quant) — download one from a GGUF-format Hugging Face repo and note its path.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set MODEL_PATH to your downloaded GGUF file
chainlit run model.py
```

On first run, the app will ask you to upload one or more documents (PDF, TXT, MD, DOCX) directly in the chat — no separate ingestion step required. You can upload more at any time by sending `upload`.

If you'd rather pre-build the index from a `Data/` folder (e.g. for a large corpus), run:

```bash
python ingest.py
```

## Configuration

All settings live in `.env` (see `.env.example`), including chunk size, retrieval `top_k`, RRF constant, and `CONTEXTUAL_HEADERS` (`heuristic` | `llm` | `off`) — `llm` mode asks your local model to write a short blurb per chunk instead of the fast heuristic header, at the cost of slower ingestion; best for small personal document sets.

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite injects a `FakeLLM` in place of a real GGUF model, so it runs without downloading multi-GB weights — it exercises ingestion, hybrid retrieval fusion, reranking, memory condensing, and citation formatting for real. It does **not** replace manually smoke-testing the live Chainlit UI with a real model, which needs an actual GGUF file to run.

## Architecture

```
ingest.py     multi-format loading, contextual chunking, builds FAISS + BM25 indexes
retrieval.py  HybridRetriever: RRF fusion + cross-encoder reranking
memory.py     condenses follow-up questions against chat history
llm.py        loads a local GGUF model via llama-cpp-python
model.py      Chainlit app: upload, retrieval, streaming, citations
config.py     all settings, loaded from .env
```
