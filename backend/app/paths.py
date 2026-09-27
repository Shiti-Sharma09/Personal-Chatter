"""Per-session filesystem paths for uploaded documents and their index.

Each chat session gets its own document set and its own FAISS/BM25
index, so a file uploaded in one conversation never leaks into another
conversation's answers or citations.
"""
import os

from app import config


def session_data_path(session_id):
    return os.path.join(config.DATA_PATH, session_id)


def session_faiss_path(session_id):
    return os.path.join(config.VECTORSTORE_ROOT, session_id, "db_faiss")


def session_bm25_path(session_id):
    return os.path.join(config.VECTORSTORE_ROOT, session_id, "bm25.pkl")
