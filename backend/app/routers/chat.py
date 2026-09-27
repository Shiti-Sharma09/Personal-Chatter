import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import db
from app.dependencies import get_llm, get_retriever
from app.memory import condense_question
from app.rag import build_prompt, format_context, format_sources

router = APIRouter()


class MessageCreate(BaseModel):
    content: str


def _title_from(content, max_len=40):
    content = content.strip().replace("\n", " ")
    return content if len(content) <= max_len else content[: max_len - 1] + "…"


@router.post("/sessions/{session_id}/messages")
def send_message(session_id: str, body: MessageCreate):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    try:
        retriever = get_retriever()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    llm = get_llm()

    history = db.list_messages(session_id)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]

    db.add_message(session_id, "user", body.content)
    if not history and session["title"] == "New chat":
        db.rename_session(session_id, _title_from(body.content))

    standalone_question = condense_question(chat_history, body.content, llm)
    retrieved = retriever.retrieve(standalone_question)
    context = format_context(retrieved)
    prompt = build_prompt(context, standalone_question)
    sources = format_sources(retrieved)

    def event_stream():
        full = ""
        for chunk in llm.stream(prompt):
            text = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
            full += text
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

        saved = db.add_message(session_id, "assistant", full.strip(), sources=sources)
        yield f"data: {json.dumps({'type': 'done', 'message_id': saved['id'], 'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
