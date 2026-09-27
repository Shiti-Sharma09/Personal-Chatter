"""Process-wide singletons for the LLM and the hybrid retriever.

Both are expensive to load (model weights, embedding model, FAISS/BM25
indexes), so they're loaded once and cached in module state rather than
per-request -- deliberately not using FastAPI's Depends DI machinery
here since there's nothing to swap at request time, just a plain cache.
"""
from app.llm import load_llm as _load_llm
from app.rag import knowledge_base_exists, load_knowledge_base

_llm = None
_retriever = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = _load_llm()
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        if not knowledge_base_exists():
            raise RuntimeError("No documents uploaded yet -- upload one before chatting.")
        _retriever = load_knowledge_base()
    return _retriever


def reset_retriever_cache():
    """Called after a new upload/ingestion so the next request picks up
    the rebuilt index instead of serving a stale in-memory one."""
    global _retriever
    _retriever = None
