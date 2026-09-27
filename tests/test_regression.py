"""End-to-end regression coverage for the new pipeline. Everything here
runs without a real GGUF model (FakeLLM stands in for llama-cpp-python) so
it's runnable in any environment with just the Python deps installed --
no multi-GB model download required. See README for the manual smoke-test
steps that do exercise a real local model.
"""
import config
import ingest
import model
from memory import condense_question
from retrieval import HybridRetriever


def test_full_pipeline_ingest_retrieve_answer(sample_data_dir, tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))

    dense_db, sparse_index = ingest.create_vector_db(data_path=sample_data_dir)

    # Real cross-encoder reranker: this is the one integration point worth
    # exercising for real (small, CPU-friendly, downloads once).
    retriever = HybridRetriever(dense_db, sparse_index)

    result = model.answer_question(retriever, fake_llm, chat_history=[], question="How do I set up the bot?")

    assert result["answer"] == fake_llm.response
    assert result["standalone_question"] == "How do I set up the bot?"  # no history -> unchanged
    assert result["retrieved"], "expected at least one retrieved chunk"
    # the prompt the fake LLM actually saw must be grounded in retrieved context
    assert "Context:" in fake_llm.prompts[-1]
    assert "Question: How do I set up the bot?" in fake_llm.prompts[-1]


def test_condense_question_uses_history_when_present(fake_llm):
    fake_llm.response = "What is the setup process for Personal Chatter?"
    history = [
        {"role": "user", "content": "Tell me about Personal Chatter."},
        {"role": "assistant", "content": "It's a RAG chatbot for personal documents."},
    ]

    standalone = condense_question(history, "How do I set it up?", fake_llm)

    assert standalone == fake_llm.response
    assert "How do I set it up?" in fake_llm.prompts[-1]
    assert "Tell me about Personal Chatter." in fake_llm.prompts[-1]


def test_condense_question_skips_llm_call_on_first_turn(fake_llm):
    standalone = condense_question([], "What is this?", fake_llm)
    assert standalone == "What is this?"
    assert fake_llm.prompts == []  # no wasted LLM call for the first turn


def test_format_sources_dedupes_and_includes_page_numbers():
    retrieved = [
        {"text": "a", "metadata": {"source": "Data/manual.pdf", "page": 0}},
        {"text": "b", "metadata": {"source": "Data/manual.pdf", "page": 0}},  # duplicate
        {"text": "c", "metadata": {"source": "Data/manual.pdf", "page": 2}},
        {"text": "d", "metadata": {"source": "Data/notes.txt"}},  # no page metadata
    ]

    sources = model.format_sources(retrieved)

    assert sources == ["manual.pdf (page 1)", "manual.pdf (page 3)", "notes.txt"]


def test_build_prompt_includes_context_and_question():
    prompt = model.build_prompt(context="Refunds take 5 business days.", question="How long do refunds take?")
    assert "Refunds take 5 business days." in prompt
    assert "How long do refunds take?" in prompt


def test_multi_turn_conversation_accumulates_history(sample_data_dir, tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))
    dense_db, sparse_index = ingest.create_vector_db(data_path=sample_data_dir)
    retriever = HybridRetriever(dense_db, sparse_index)

    history = []
    first = model.answer_question(retriever, fake_llm, history, "What did the team ship in Q3?")
    history.append({"role": "user", "content": "What did the team ship in Q3?"})
    history.append({"role": "assistant", "content": first["answer"]})

    fake_llm.response = "The latency stayed under 2 seconds."
    second = model.answer_question(retriever, fake_llm, history, "And what was the latency?")

    # the follow-up must have gone through condensation against history,
    # i.e. the LLM was invoked twice more (condense + answer) with the
    # accumulated history present in the condense prompt
    condense_prompt = fake_llm.prompts[-2]
    assert "What did the team ship in Q3?" in condense_prompt
    assert "And what was the latency?" in condense_prompt
    assert second["answer"] == "The latency stayed under 2 seconds."
