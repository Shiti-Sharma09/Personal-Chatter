"""FastAPI-level regression tests: sessions, uploads, and the streaming
chat endpoint, wired end-to-end against real ingestion/retrieval and a
temp SQLite DB, with FakeLLM standing in for the GGUF model."""
import io
import os

import pytest
from fastapi.testclient import TestClient

from app import config, db, dependencies
from app.main import app


@pytest.fixture
def client(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "Data"))
    monkeypatch.setattr(config, "VECTORSTORE_ROOT", str(tmp_path / "vectorstore"))

    dependencies._llm = fake_llm
    dependencies._retrievers = {}
    db.init_db()

    with TestClient(app) as test_client:
        yield test_client

    dependencies._llm = None
    dependencies._retrievers = {}


def _upload(client, session_id, filename, content):
    return client.post(
        f"/api/sessions/{session_id}/documents/upload",
        files={"files": (filename, io.BytesIO(content), "text/markdown")},
    )


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
    session = client.post("/api/sessions", json={}).json()

    resp = _upload(client, session["id"], "policy.md", b"# Refund Policy\nRefunds take 5 business days to process.\n")
    assert resp.status_code == 200
    assert resp.json()["ingested"] == ["policy.md"]

    docs = client.get(f"/api/sessions/{session['id']}/documents").json()
    assert any(d["name"] == "policy.md" for d in docs)

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


def test_documents_are_isolated_per_session(client):
    """Regression test for a real bug: uploads used to go into one shared
    knowledge base, so a document uploaded in one chat leaked into another
    chat's retrieved sources. Each session must only ever see its own docs."""
    session_a = client.post("/api/sessions", json={}).json()
    session_b = client.post("/api/sessions", json={}).json()

    _upload(client, session_a["id"], "trip.md", b"# Trip\nDay 4 in Vietnam: Cu Chi tunnels tour.\n")
    _upload(client, session_b["id"], "recipe.md", b"# Recipe\nBake the bread for 40 minutes at 220C.\n")

    docs_a = client.get(f"/api/sessions/{session_a['id']}/documents").json()
    docs_b = client.get(f"/api/sessions/{session_b['id']}/documents").json()
    assert [d["name"] for d in docs_a] == ["trip.md"]
    assert [d["name"] for d in docs_b] == ["recipe.md"]

    with client.stream(
        "POST", f"/api/sessions/{session_a['id']}/messages", json={"content": "what happens on day 4?"}
    ) as resp:
        events = "".join(resp.iter_lines())
    assert "trip.md" in events
    assert "recipe.md" not in events


def test_upload_rejects_oversized_file_and_cleans_up(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    session = client.post("/api/sessions", json={}).json()

    resp = _upload(client, session["id"], "big.md", b"x" * 2000)
    assert resp.status_code == 400

    assert client.get(f"/api/sessions/{session['id']}/documents").json() == []
    data_dir = os.path.join(str(config.DATA_PATH), session["id"])
    assert not os.path.exists(os.path.join(data_dir, "big.md"))


class _FailingLLM:
    """Simulates the model erroring partway through generation (context
    overflow, crash, etc.) to verify the SSE stream reports it cleanly
    instead of just breaking the connection -- see routers/chat.py."""

    def invoke(self, messages, stop=None):
        return "unused on a first turn (no history to condense)"

    def stream(self, messages, stop=None):
        yield "partial "
        yield "answer"
        raise RuntimeError("model crashed")


def test_chat_stream_reports_error_and_persists_partial_answer(client):
    session = client.post("/api/sessions", json={}).json()
    _upload(client, session["id"], "note.md", b"# Note\nSome content.\n")
    dependencies._llm = _FailingLLM()

    with client.stream(
        "POST", f"/api/sessions/{session['id']}/messages", json={"content": "hello"}
    ) as resp:
        events = [line for line in resp.iter_lines() if line.startswith("data:")]

    assert any('"type": "error"' in e for e in events)
    assert not any('"type": "done"' in e for e in events)

    messages = client.get(f"/api/sessions/{session['id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "partial answer"


def test_delete_session_cleans_up_documents(client):
    session = client.post("/api/sessions", json={}).json()
    _upload(client, session["id"], "note.md", b"# Note\nSome content.\n")

    data_dir = os.path.join(str(config.DATA_PATH), session["id"])
    assert os.path.isdir(data_dir)

    resp = client.delete(f"/api/sessions/{session['id']}")
    assert resp.status_code == 200
    assert client.get(f"/api/sessions/{session['id']}/messages").status_code == 404
    assert not os.path.isdir(data_dir)
