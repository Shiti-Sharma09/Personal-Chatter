"""Chainlit entrypoint.

Upgrades over the original:
  - real in-chat file upload (the README always claimed this; the old
    code only ever read a pre-populated Data/ folder via an offline script)
  - hybrid dense+sparse retrieval with reranking instead of bare
    `db.as_retriever(k=2)`
  - conversational memory: follow-up questions are condensed against
    chat history before retrieval
  - true token streaming via a background thread instead of a Langchain
    callback handler paired with a backend that didn't really support it
  - citations include page numbers and are deduped
"""
import asyncio
import os
import shutil

import chainlit as cl

import config
import ingest
from llm import load_llm
from memory import condense_question
from retrieval import HybridRetriever

WELCOME_MESSAGE = "Hi! I'm Personal Chatter \U0001F4AC — ask me anything about your uploaded documents."

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
    """Dedup source citations, surfacing page numbers when available
    instead of the original's bare, occasionally-duplicated file paths."""
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
    from ingest import SparseIndex

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    # allow_dangerous_deserialization is safe here specifically because
    # this codebase is the only writer of DB_FAISS_PATH (see
    # ingest.create_vector_db) -- we only ever load pickles we produced
    # ourselves, never an arbitrary/untrusted path.
    dense_db = FAISS.load_local(config.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    sparse_index = SparseIndex.load(config.DB_BM25_PATH)
    return HybridRetriever(dense_db, sparse_index)


def answer_question(retriever, llm, chat_history, question):
    """Pure retrieval+prompt pipeline shared by the interactive app and
    the regression tests (no streaming, no Chainlit dependency)."""
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


async def _stream_llm(llm, prompt, msg):
    """Stream tokens from the local (sync/blocking) LLM into a Chainlit
    message without blocking the event loop, via a background thread."""
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _produce():
        try:
            for chunk in llm.stream(prompt):
                text = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
                loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as exc:  # surfaced to the consumer below, not swallowed
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _produce)

    full = ""
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        full += chunk
        await msg.stream_token(chunk)
    return full


async def _ingest_uploads(files):
    os.makedirs(config.DATA_PATH, exist_ok=True)
    for f in files:
        dest = os.path.join(config.DATA_PATH, os.path.basename(getattr(f, "name", None) or f.path))
        shutil.copyfile(f.path, dest)
    await cl.make_async(ingest.create_vector_db)()


@cl.on_chat_start
async def start():
    if not knowledge_base_exists():
        files = None
        while not files:
            files = await cl.AskFileMessage(
                content="Upload one or more documents (PDF, TXT, MD, DOCX) to start chatting with them.",
                accept=list(config.SUPPORTED_EXTENSIONS),
                max_files=10,
                max_size_mb=50,
                timeout=600,
            ).send()

        indexing_msg = cl.Message(content=f"Indexing {len(files)} file(s)…")
        await indexing_msg.send()
        await _ingest_uploads(files)
        indexing_msg.content = "Indexing complete!"
        await indexing_msg.update()

    cl.user_session.set("retriever", load_knowledge_base())
    cl.user_session.set("llm", load_llm())
    cl.user_session.set("chat_history", [])

    await cl.Message(content=WELCOME_MESSAGE + ' (Send "upload" anytime to add more documents.)').send()


@cl.on_message
async def main(message: cl.Message):
    if message.content.strip().lower() == "upload":
        files = await cl.AskFileMessage(
            content="Upload additional documents.",
            accept=list(config.SUPPORTED_EXTENSIONS),
            max_files=10,
            max_size_mb=50,
            timeout=600,
        ).send()
        if files:
            status = cl.Message(content=f"Indexing {len(files)} more file(s)…")
            await status.send()
            await _ingest_uploads(files)
            cl.user_session.set("retriever", load_knowledge_base())
            status.content = "Indexing complete! Ask away."
            await status.update()
        return

    retriever = cl.user_session.get("retriever")
    llm = cl.user_session.get("llm")
    chat_history = cl.user_session.get("chat_history", [])

    try:
        standalone_question = condense_question(chat_history, message.content, llm)
        retrieved = retriever.retrieve(standalone_question)
        context = format_context(retrieved)
        prompt = build_prompt(context, standalone_question)

        msg = cl.Message(content="")
        await msg.send()
        answer = await _stream_llm(llm, prompt, msg)

        sources = format_sources(retrieved)
        msg.content = answer + ("\n\nSources:\n" + "\n".join(sources) if sources else "\n\nNo sources found.")
        await msg.update()

        chat_history.append({"role": "user", "content": message.content})
        chat_history.append({"role": "assistant", "content": answer})
        cl.user_session.set("chat_history", chat_history)
    except Exception as exc:
        print(f"Error processing message: {exc}")
        await cl.Message(content="Sorry, something went wrong answering that.").send()
