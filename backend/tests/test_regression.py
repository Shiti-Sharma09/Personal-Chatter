"""End-to-end regression coverage for the new pipeline. Everything here
runs without a real GGUF model (FakeLLM stands in for llama-cpp-python) so
it's runnable in any environment with just the Python deps installed --
no multi-GB model download required. See README for the manual smoke-test
steps that do exercise a real local model.
"""
from app import config, ingest, rag
from app.memory import condense_question
from app.retrieval import HybridRetriever
from conftest import flatten_prompt


def test_full_pipeline_ingest_retrieve_answer(sample_data_dir, tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))

    dense_db, sparse_index = ingest.create_vector_db(data_path=sample_data_dir)

    # Real cross-encoder reranker: this is the one integration point worth
    # exercising for real (small, CPU-friendly, downloads once).
    retriever = HybridRetriever(dense_db, sparse_index)

    result = rag.answer_question(retriever, fake_llm, chat_history=[], question="How do I set up the bot?")

    assert result["answer"] == fake_llm.response
    assert result["standalone_question"] == "How do I set up the bot?"  # no history -> unchanged
    assert result["retrieved"], "expected at least one retrieved chunk"
    # the prompt the fake LLM actually saw must be grounded in retrieved context
    last_prompt = flatten_prompt(fake_llm.prompts[-1])
    assert "Context:" in last_prompt
    assert "Question: How do I set up the bot?" in last_prompt


def test_condense_question_uses_history_when_present(fake_llm):
    fake_llm.response = "What is the setup process for Personal Chatter?"
    history = [
        {"role": "user", "content": "Tell me about Personal Chatter."},
        {"role": "assistant", "content": "It's a RAG chatbot for personal documents."},
    ]

    standalone = condense_question(history, "How do I set it up?", fake_llm)

    assert standalone == fake_llm.response
    last_prompt = flatten_prompt(fake_llm.prompts[-1])
    assert "How do I set it up?" in last_prompt
    assert "Tell me about Personal Chatter." in last_prompt


def test_condense_question_skips_llm_call_on_first_turn(fake_llm):
    standalone = condense_question([], "What is this?", fake_llm)
    assert standalone == "What is this?"
    assert fake_llm.prompts == []  # no wasted LLM call for the first turn


def test_condense_question_falls_back_when_model_echoes_prompt_scaffolding(fake_llm):
    """Regression test: observed live against a real 3B instruct model,
    which crammed the prompt's own instructions onto the single line it's
    allowed ("Rewritten standalone question: ... Answer only with the
    above format.") instead of a real rewritten question. That garbage
    must never reach retrieval or the next prompt -- falling back to the
    original question is strictly safer than trusting it."""
    fake_llm.response = "Rewritten standalone question: what activities happen on day 4? Answer only with the above format."
    history = [
        {"role": "user", "content": "What happens on day 4?"},
        {"role": "assistant", "content": "The itinerary includes a free afternoon."},
    ]

    standalone = condense_question(history, "and the evening?", fake_llm)

    assert standalone == "and the evening?"


def test_format_sources_dedupes_and_includes_page_numbers():
    retrieved = [
        {"text": "a", "metadata": {"source": "Data/manual.pdf", "page": 0}},
        {"text": "b", "metadata": {"source": "Data/manual.pdf", "page": 0}},  # duplicate
        {"text": "c", "metadata": {"source": "Data/manual.pdf", "page": 2}},
        {"text": "d", "metadata": {"source": "Data/notes.txt"}},  # no page metadata
    ]

    sources = rag.format_sources(retrieved)

    assert sources == ["manual.pdf (page 1)", "manual.pdf (page 3)", "notes.txt"]


def test_build_messages_includes_context_and_question():
    messages = rag.build_messages(context="Refunds take 5 business days.", question="How long do refunds take?")
    flattened = flatten_prompt(messages)
    assert "Refunds take 5 business days." in flattened
    assert "How long do refunds take?" in flattened
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_multi_turn_conversation_accumulates_history(sample_data_dir, tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))
    dense_db, sparse_index = ingest.create_vector_db(data_path=sample_data_dir)
    retriever = HybridRetriever(dense_db, sparse_index)

    history = []
    first = rag.answer_question(retriever, fake_llm, history, "What did the team ship in Q3?")
    history.append({"role": "user", "content": "What did the team ship in Q3?"})
    history.append({"role": "assistant", "content": first["answer"]})

    fake_llm.response = "The latency stayed under 2 seconds."
    second = rag.answer_question(retriever, fake_llm, history, "And what was the latency?")

    # the follow-up must have gone through condensation against history,
    # i.e. the LLM was invoked twice more (condense + answer) with the
    # accumulated history present in the condense prompt
    condense_prompt = flatten_prompt(fake_llm.prompts[-2])
    assert "What did the team ship in Q3?" in condense_prompt
    assert "And what was the latency?" in condense_prompt
    assert second["answer"] == "The latency stayed under 2 seconds."
