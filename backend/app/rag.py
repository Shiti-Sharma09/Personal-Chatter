"""Prompt construction, citation formatting, and knowledge-base loading --
the pure, framework-free pieces of the RAG pipeline, shared by the API
routers and the test suite (no FastAPI or DB dependency here)."""
import os

from app import config
from app.retrieval import HybridRetriever

custom_prompt_template = """You are a helpful assistant answering questions using only the provided context.
If the answer isn't in the context, say you don't know — don't make one up.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(context, question):
    return custom_prompt_template.format(context=context, question=question)


def format_context(retrieved):
    return "\n\n---\n\n".join(item["text"] for item in retrieved)


def format_sources(retrieved):
    """Dedup source citations, surfacing page numbers when available."""
    seen = []
    for item in retrieved:
        metadata = item["metadata"]
        source = os.path.basename(metadata.get("source", "unknown"))
        page = metadata.get("page")
        label = f"{source} (page {page + 1})" if page is not None else source
        if label not in seen:
            seen.append(label)
    return seen


def knowledge_base_exists():
    return os.path.isdir(config.DB_FAISS_PATH) and os.path.isfile(config.DB_BM25_PATH)


def load_knowledge_base():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    from app.ingest import SparseIndex

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    # allow_dangerous_deserialization is safe here specifically because
    # this codebase is the only writer of DB_FAISS_PATH (see
    # ingest.create_vector_db) -- we only ever load pickles we produced
    # ourselves, never an arbitrary/untrusted path.
    dense_db = FAISS.load_local(config.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    sparse_index = SparseIndex.load(config.DB_BM25_PATH)
    return HybridRetriever(dense_db, sparse_index)


def answer_question(retriever, llm, chat_history, question):
    """Pure retrieval+prompt pipeline: condense -> retrieve -> prompt ->
    invoke. Non-streaming; used directly by tests, and mirrored by the
    streaming SSE path in routers/chat.py for the live API."""
    from app.memory import condense_question

    standalone_question = condense_question(chat_history, question, llm)
    retrieved = retriever.retrieve(standalone_question)
    context = format_context(retrieved)
    prompt = build_prompt(context, standalone_question)
    raw_answer = llm.invoke(prompt)
    answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)
    return {
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": format_sources(retrieved),
        "retrieved": retrieved,
    }
