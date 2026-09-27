"""Central configuration, loaded from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    return int(os.getenv(name, default))


def _float(name, default):
    return float(os.getenv(name, default))


# Root directory under which each chat session gets its own subfolder
# for uploaded documents (Data/<session_id>/) -- see app/paths.py. Kept
# non-recursive by ingest.load_documents, so files placed directly in
# DATA_PATH (not under a session subfolder) are invisible to per-session
# chat and only used by the standalone `python -m app.ingest` CLI path.
DATA_PATH = os.getenv("DATA_PATH", "Data/")
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
DB_BM25_PATH = os.getenv("DB_BM25_PATH", "vectorstore/bm25.pkl")
# Root under which each session's FAISS/BM25 indexes live (vectorstore/<session_id>/...).
VECTORSTORE_ROOT = os.getenv("VECTORSTORE_ROOT", "vectorstore/sessions")
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "sessions.db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MODEL_PATH = os.getenv("MODEL_PATH", "models/llama-3-8b-instruct.Q4_K_M.gguf")
LLM_CONTEXT_WINDOW = _int("LLM_CONTEXT_WINDOW", 4096)
LLM_MAX_NEW_TOKENS = _int("LLM_MAX_NEW_TOKENS", 512)
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.3)
LLM_GPU_LAYERS = _int("LLM_GPU_LAYERS", 0)
LLM_REPEAT_PENALTY = _float("LLM_REPEAT_PENALTY", 1.15)
# Explicit chat-template format name (e.g. "chatml", "llama-3",
# "mistral-instruct") for GGUF files that don't embed their own
# template. Leave unset to auto-detect from the model file.
LLM_CHAT_FORMAT = os.getenv("LLM_CHAT_FORMAT", "")

CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# Contextual retrieval: prepend each chunk with a short header describing
# where it sits in the source document before embedding it. The heuristic
# mode (filename + nearest heading + page) is fast and needs no LLM calls;
# the LLM mode asks the local model to write a 1-2 sentence blurb per chunk
# (per Anthropic's "contextual retrieval" technique) — slower, better recall,
# best reserved for small personal document sets.
CONTEXTUAL_HEADERS = os.getenv("CONTEXTUAL_HEADERS", "heuristic")  # "heuristic" | "llm" | "off"

# Hybrid retrieval
DENSE_TOP_K = _int("DENSE_TOP_K", 10)
SPARSE_TOP_K = _int("SPARSE_TOP_K", 10)
RRF_K = _int("RRF_K", 60)
RERANK_TOP_N = _int("RERANK_TOP_N", 4)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
