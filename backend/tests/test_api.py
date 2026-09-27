"""FastAPI-level regression tests: sessions, uploads, and the streaming
chat endpoint, wired end-to-end against real ingestion/retrieval and a
temp SQLite DB, with FakeLLM standing in for the GGUF model."""
import io

import pytest
from fastapi.testclient import TestClient

from app import config, db, dependencies
from app.main import app


@pytest.fixture
def client(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "Data"))
    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))

    dependencies._llm = fake_llm
    dependencies._retriever = None
    db.init_db()

    with TestClient(app) as test_client:
        yield test_client

    dependencies._llm = None
    dependencies._retriever = None


def test_create_and_list_sessions(client):
    created = client.post("/api/sessions", json={"title": "My chat"}).json()
    assert created["title"] == "My chat"

    sessions = client.get("/api/sessions").json()
    assert any(s["id"] == created["id"] for s in sessions)


def test_get_messages_for_unknown_session_404s(client):
    resp = client.get("/api/sessions/does-not-exist/messages")
    assert resp.status_code == 404


def test_chat_without_documents_returns_400(client):
    session = client.post("/api/sessions", json={}).json()
    resp = client.post(f"/api/sessions/{session['id']}/messages", json={"content": "hello"})
    assert resp.status_code == 400


def test_upload_then_chat_streams_answer_and_persists_history(client):
    doc = b"# Refund Policy\nRefunds take 5 business days to process.\n"
    resp = client.post(
        "/api/documents/upload",
        files={"files": ("policy.md", io.BytesIO(doc), "text/markdown")},
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == ["policy.md"]

    docs = client.get("/api/documents").json()
    assert any(d["name"] == "policy.md" for d in docs)

    session = client.post("/api/sessions", json={}).json()

    with client.stream(
        "POST", f"/api/sessions/{session['id']}/messages", json={"content": "How long do refunds take?"}
    ) as resp:
        assert resp.status_code == 200
        events = [line for line in resp.iter_lines() if line.startswith("data:")]
    assert events, "expected at least one SSE event"
    assert any('"type": "done"' in e for e in events)

    messages = client.get(f"/api/sessions/{session['id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "How long do refunds take?"

    # first message should have auto-titled the session
    sessions = client.get("/api/sessions").json()
    updated = next(s for s in sessions if s["id"] == session["id"])
    assert updated["title"] == "How long do refunds take?"


def test_delete_session(client):
    session = client.post("/api/sessions", json={}).json()
    resp = client.delete(f"/api/sessions/{session['id']}")
    assert resp.status_code == 200
    assert client.get(f"/api/sessions/{session['id']}/messages").status_code == 404
