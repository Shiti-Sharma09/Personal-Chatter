"""Build the hybrid (dense + sparse) index over Data/.

Upgrades over the original naive pipeline:
  - multi-format loading (pdf, txt, md, docx) instead of PDF-only
  - contextual chunk headers (Anthropic-style "contextual retrieval"):
    each chunk is prepended with a short header naming its source
    document, page, and nearest section heading, which measurably
    improves retrieval precision over bare chunk text
  - builds both a FAISS dense index and a BM25 sparse index, so
    retrieval.py can fuse them (hybrid search) instead of relying on
    embeddings alone
"""
import os
import pickle
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app import config

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def load_documents(data_path=None):
    """Load every supported file under data_path via its matching loader."""
    # Resolved at call time, not as a default-argument value: a default
    # of config.DATA_PATH would freeze in whatever DATA_PATH was when
    # this module was first imported, silently ignoring later env/config
    # changes (e.g. a test monkeypatching config.DATA_PATH, or a real
    # .env reload).
    data_path = data_path if data_path is not None else config.DATA_PATH
    documents = []
    if not os.path.isdir(data_path):
        return documents

    for filename in sorted(os.listdir(data_path)):
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            continue
        file_path = os.path.join(data_path, filename)
        loader = loader_cls(file_path)
        documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(documents)


def _nearest_heading(full_text, start_index):
    """Find the closest markdown-style heading preceding start_index."""
    heading = None
    for line in full_text[:start_index].splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            heading = match.group(2).strip()
    return heading


def _build_header(chunk, full_text_by_source):
    source = os.path.basename(chunk.metadata.get("source", "unknown"))
    parts = [f"Document: {source}"]

    page = chunk.metadata.get("page")
    if page is not None:
        parts.append(f"Page: {page + 1}")

    full_text = full_text_by_source.get(chunk.metadata.get("source"))
    start_index = chunk.metadata.get("start_index")
    if full_text is not None and start_index is not None:
        heading = _nearest_heading(full_text, start_index)
        if heading:
            parts.append(f"Section: {heading}")

    return " | ".join(parts)


def _llm_context(llm, full_text, chunk_text):
    """Ask the local LLM for a 1-2 sentence blurb situating this chunk in
    the document (richer variant of contextual retrieval). Best used on
    small personal document sets given local CPU inference cost."""
    prompt = (
        "Document excerpt:\n"
        f"{full_text[:2000]}\n\n"
        "Chunk to contextualize:\n"
        f"{chunk_text}\n\n"
        "In 1-2 sentences, describe what this chunk is about and how it "
        "relates to the document, to help retrieve it later. Answer only "
        "with the description."
    )
    return llm.invoke(prompt).strip()


def add_contextual_headers(chunks, documents, mode=None, llm=None):
    """Prepend each chunk's text with a short context header. `mode` is
    'heuristic' (default, fast, no model needed), 'llm' (richer, slower),
    or 'off' (leave chunks untouched)."""
    mode = mode if mode is not None else config.CONTEXTUAL_HEADERS
    if mode == "off":
        return chunks

    full_text_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source")
        full_text_by_source.setdefault(source, "")
        full_text_by_source[source] += doc.page_content + "\n"

    for chunk in chunks:
        if mode == "llm" and llm is not None:
            header = _llm_context(llm, full_text_by_source.get(chunk.metadata.get("source"), ""), chunk.page_content)
        else:
            header = _build_header(chunk, full_text_by_source)
        chunk.metadata["context_header"] = header
        chunk.page_content = f"[{header}]\n{chunk.page_content}"

    return chunks


class SparseIndex:
    """Lightweight BM25 index wrapper that keeps chunk text/metadata
    alongside the scorer so it can be persisted and queried standalone,
    the way FAISS keeps its own docstore."""

    def __init__(self, texts, metadatas):
        self.texts = texts
        self.metadatas = metadatas
        self._tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self._tokenized)

    def query(self, query_text, k):
        scores = self.bm25.get_scores(query_text.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.texts[i], self.metadatas[i], scores[i]) for i in ranked]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, path):
        # Same trust model as FAISS.load_local's allow_dangerous_deserialization
        # (see rag.py): safe because this class is the only writer of these
        # paths (see save() above and create_vector_db() below) -- never an
        # arbitrary/untrusted path.
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["texts"], data["metadatas"])


def create_vector_db(data_path=None, contextual_mode=None, llm=None, faiss_path=None, bm25_path=None):
    data_path = data_path if data_path is not None else config.DATA_PATH
    contextual_mode = contextual_mode if contextual_mode is not None else config.CONTEXTUAL_HEADERS
    faiss_path = faiss_path if faiss_path is not None else config.DB_FAISS_PATH
    bm25_path = bm25_path if bm25_path is not None else config.DB_BM25_PATH

    documents = load_documents(data_path)
    if not documents:
        raise ValueError(f"No supported documents found under '{data_path}' ({config.SUPPORTED_EXTENSIONS})")

    chunks = split_documents(documents)
    chunks = add_contextual_headers(chunks, documents, mode=contextual_mode, llm=llm)

    embeddings = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)
    dense_db = FAISS.from_documents(chunks, embeddings)
    dense_db.save_local(faiss_path)

    sparse_index = SparseIndex(
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    sparse_index.save(bm25_path)

    return dense_db, sparse_index


if __name__ == "__main__":
    create_vector_db()
