from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db

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
    return {"ok": True}
