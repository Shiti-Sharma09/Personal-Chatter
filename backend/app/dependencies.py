"""Process-wide singletons for the LLM and per-session hybrid retrievers.

The LLM is one shared model instance (loading it is expensive and it's
stateless across requests). Retrievers are cached per session_id instead
of as a single global, since each chat session has its own document set
and index (see app/paths.py) -- a single shared retriever would leak one
conversation's uploaded documents into another's answers.
"""
from app.llm import load_llm as _load_llm
from app import paths
from app.rag import knowledge_base_exists, load_knowledge_base

_llm = None
_retrievers = {}


def get_llm():
    global _llm
    if _llm is None:
        _llm = _load_llm()
    return _llm


def get_retriever(session_id):
    if session_id not in _retrievers:
        faiss_path = paths.session_faiss_path(session_id)
        bm25_path = paths.session_bm25_path(session_id)
        if not knowledge_base_exists(faiss_path, bm25_path):
            raise RuntimeError("No documents uploaded in this chat yet -- upload one before chatting.")
        _retrievers[session_id] = load_knowledge_base(faiss_path, bm25_path)
    return _retrievers[session_id]


def reset_retriever_cache(session_id):
    """Called after a session's documents are (re)uploaded, or the
    session is deleted, so a stale in-memory index isn't served."""
    _retrievers.pop(session_id, None)
