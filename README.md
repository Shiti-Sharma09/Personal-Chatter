# Personal Chatter

A local, open-source RAG chatbot: upload your own documents and chat with them, grounded in citations, across saved conversations. Runs entirely offline — no API keys, no data leaving your machine.

## Architecture

```
backend/            FastAPI + SQLite
  app/
    ingest.py        multi-format loading, contextual chunking, builds FAISS + BM25 indexes
    retrieval.py      HybridRetriever: RRF fusion + cross-encoder reranking
    memory.py          condenses follow-up questions against chat history
    llm.py              loads a local GGUF model via llama-cpp-python
    rag.py               prompt construction, citation formatting, KB loading
    db.py                 session/message persistence (stdlib sqlite3)
    dependencies.py        process-wide LLM/retriever singletons
    routers/                 sessions.py, chat.py (SSE streaming), documents.py (upload)
    main.py                    FastAPI app wiring
  config.py            all settings, loaded from .env
  tests/                pytest suite (ingestion, retrieval, memory, API)

frontend/            React 19 + TypeScript + Vite + Tailwind CSS v4
  src/
    api.ts             typed fetch client + SSE stream parsing
    components/         Sidebar, ChatWindow, MessageBubble, UploadDropzone
    App.tsx               session/chat state
```

Sessions and message history live in a single SQLite file (`backend/sessions.db`) — no separate database server to run or maintain, and no ORM/migration tooling: just a file, which is the right amount of infrastructure for a single-user local app.

## What changed (naive RAG → advanced RAG)

The original version of this project was a straightforward 2023-era LangChain quickstart clone (a single Chainlit script). This upgrade closes the gap between that and what's actually state-of-the-art in RAG today, and replaces the prototype UI with a real client/server app:

| | Before | Now |
|---|---|---|
| Retrieval | Dense-only FAISS lookup, hardcoded `k=2` | **Hybrid search**: dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion |
| Ranking | Whatever the embedding model returned, unranked further | **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the fused candidates |
| Chunking | Fixed-size splitting only | Fixed-size splitting **plus contextual chunk headers** (source, page, section) prepended before embedding — a cheap form of Anthropic's "contextual retrieval," shown to cut retrieval-miss rate |
| Conversation | Every message treated as a fresh, context-free query, nothing persisted | **Conversational memory** — follow-ups condensed into standalone questions — **and saved sessions** in SQLite, browsable in a sidebar |
| File upload | README claimed it; code only read a pre-populated `Data/` folder via an offline script | **Real in-app upload**, drag-and-drop, plus PDF/TXT/MD/DOCX support (was PDF-only) |
| Citations | Raw file paths, naive dedup | `filename (page N)`, deduped, shown as pills under each answer |
| LLM runtime | `CTransformers` + GGML (deprecated ~2 years ago) | `llama-cpp-python` + GGUF |
| Frontend | Chainlit (prototyping tool, not meant for a polished product UI) | React 19 + TypeScript + Tailwind, served by FastAPI |
| Streaming | Langchain callback paired with a backend that didn't really support token streaming | Real token streaming over Server-Sent Events into the chat UI |
| Tests | None | `pytest` suite (ingestion, hybrid fusion, reranking, memory, full pipeline, FastAPI endpoints) |

## Prerequisites

- Python 3.10+
- Node.js 20+
- A local GGUF model (e.g. a Llama-3 or Mistral instruct quant) — download one from a GGUF-format Hugging Face repo and note its path.

## Setup

**Backend:**

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env: set MODEL_PATH to your downloaded GGUF file
uvicorn app.main:app --reload
```

**Frontend** (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`). The dev server proxies `/api` requests to the backend on `:8000`.

On first launch, upload one or more documents (PDF, TXT, MD, DOCX) — no separate ingestion step required. Add more any time via the sidebar's Documents panel.

For production, build the frontend and let FastAPI serve it from the same origin:

```bash
cd frontend && npm run build   # writes frontend/dist
cd ../backend && uvicorn app.main:app
```

If you'd rather pre-build the index from a `Data/` folder (e.g. for a large corpus) instead of uploading through the UI:

```bash
cd backend && python -m app.ingest
```

## Configuration

All backend settings live in `backend/.env` (see `.env.example`), including chunk size, retrieval `top_k`, RRF constant, the SQLite path, and `CONTEXTUAL_HEADERS` (`heuristic` | `llm` | `off`) — `llm` mode asks your local model to write a short blurb per chunk instead of the fast heuristic header, at the cost of slower ingestion; best for small personal document sets.

## Security

There is no authentication or authorization anywhere in this app — it's designed to run on `localhost` for one person. Anyone who can reach the backend's port can create/delete sessions, read chat history, and upload documents. Don't expose it to the public internet or an untrusted network without adding an auth layer in front of it.

## Testing

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

The test suite injects a `FakeLLM` in place of a real GGUF model, so it runs without downloading multi-GB weights — it exercises ingestion, hybrid retrieval fusion, reranking, memory condensing, citation formatting, and the FastAPI endpoints (sessions, upload, streaming chat) for real. It does **not** replace manually smoke-testing the live app with a real model, which needs an actual GGUF file to run.

```bash
cd frontend
npm run build   # type-checks and builds; fails on TS errors
```
