import os
import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db, paths
from app.dependencies import reset_retriever_cache

router = APIRouter()


class SessionCreate(BaseModel):
    title: str | None = None


@router.post("")
def create_session(body: SessionCreate | None = None):
    title = (body.title if body and body.title else None) or "New chat"
    return db.create_session(title)


@router.get("")
def list_sessions():
    return db.list_sessions()


@router.get("/{session_id}/messages")
def get_messages(session_id: str):
    if db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return db.list_messages(session_id)


@router.delete("/{session_id}")
def remove_session(session_id: str):
    if db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete_session(session_id)
    reset_retriever_cache(session_id)
    # Best-effort: a deleted session's uploaded documents and index are
    # no longer reachable through the API, so don't leave them on disk.
    shutil.rmtree(paths.session_data_path(session_id), ignore_errors=True)
    vectorstore_dir = os.path.dirname(paths.session_faiss_path(session_id))
    shutil.rmtree(vectorstore_dir, ignore_errors=True)
    return {"ok": True}
