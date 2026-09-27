from app import ingest


def test_load_documents_multi_format(sample_data_dir):
    docs = ingest.load_documents(sample_data_dir)
    sources = {d.metadata["source"] for d in docs}
    assert len(docs) == 2
    assert any(s.endswith("guide.md") for s in sources)
    assert any(s.endswith("notes.txt") for s in sources)


def test_load_documents_skips_unsupported(tmp_path):
    (tmp_path / "readme.exe").write_bytes(b"not a real doc")
    docs = ingest.load_documents(str(tmp_path))
    assert docs == []


def test_load_documents_missing_dir(tmp_path):
    docs = ingest.load_documents(str(tmp_path / "does-not-exist"))
    assert docs == []


def test_split_documents_produces_multiple_chunks(sample_data_dir):
    docs = ingest.load_documents(sample_data_dir)
    chunks = ingest.split_documents(docs, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > len(docs)
    assert all("start_index" in c.metadata for c in chunks)


def test_contextual_headers_heuristic_adds_document_and_heading(sample_data_dir):
    docs = ingest.load_documents(sample_data_dir)
    chunks = ingest.split_documents(docs, chunk_size=120, chunk_overlap=20)
    chunks = ingest.add_contextual_headers(chunks, docs, mode="heuristic")

    md_chunks = [c for c in chunks if c.metadata["source"].endswith("guide.md")]
    assert md_chunks, "expected at least one chunk from guide.md"
    for chunk in md_chunks:
        assert chunk.page_content.startswith("[Document: guide.md")
        assert "context_header" in chunk.metadata

    # A chunk taken from under "## Usage" should have that section recorded.
    usage_chunk = next(c for c in md_chunks if "Upload a document" in c.page_content)
    assert "Section: Usage" in usage_chunk.metadata["context_header"]


def test_contextual_headers_off_leaves_chunks_untouched(sample_data_dir):
    docs = ingest.load_documents(sample_data_dir)
    chunks = ingest.split_documents(docs, chunk_size=120, chunk_overlap=20)
    original = [c.page_content for c in chunks]
    chunks = ingest.add_contextual_headers(chunks, docs, mode="off")
    assert [c.page_content for c in chunks] == original


def test_sparse_index_roundtrip(tmp_path):
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    metadatas = [{"source": "a.txt"}, {"source": "b.txt"}]
    index = ingest.SparseIndex(texts, metadatas)
    path = str(tmp_path / "bm25.pkl")
    index.save(path)

    loaded = ingest.SparseIndex.load(path)
    results = loaded.query("quick fox", k=1)
    assert results[0][0] == "the quick brown fox"


def test_create_vector_db_builds_dense_and_sparse_indexes(sample_data_dir, tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "DB_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setattr(config, "DB_BM25_PATH", str(tmp_path / "bm25.pkl"))

    dense_db, sparse_index = ingest.create_vector_db(data_path=sample_data_dir)

    assert dense_db.similarity_search("What is this document about?", k=1)
    assert sparse_index.query("hybrid retrieval upgrade", k=1)
    import os

    assert os.path.isdir(config.DB_FAISS_PATH)
    assert os.path.isfile(config.DB_BM25_PATH)


def test_create_vector_db_raises_on_empty_dir(tmp_path):
    empty_dir = tmp_path / "Data"
    empty_dir.mkdir()
    try:
        ingest.create_vector_db(data_path=str(empty_dir))
        assert False, "expected ValueError for empty data dir"
    except ValueError:
        pass
