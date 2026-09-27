from retrieval import HybridRetriever, reciprocal_rank_fusion


def test_rrf_boosts_candidates_present_in_both_lists():
    dense = [("A", {"source": "a"}), ("B", {"source": "b"}), ("C", {"source": "c"})]
    sparse = [("C", {"source": "c"}), ("D", {"source": "d"})]

    fused = reciprocal_rank_fusion([dense, sparse])

    # C ranks 3rd in dense but 1st in sparse -> combined score should put
    # it ahead of A, which only ever appears in one list at rank 0.
    fused_texts = [text for text, _metadata, _score in fused]
    assert fused_texts[0] == "C"


def test_rrf_dedupes_using_source_and_start_index():
    dense = [("same text, different chunk id", {"source": "a", "start_index": 0})]
    sparse = [("same text, different chunk id", {"source": "a", "start_index": 0})]

    fused = reciprocal_rank_fusion([dense, sparse])
    assert len(fused) == 1


class _FakeDoc:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


class _FakeDenseDB:
    def __init__(self, ranked_docs):
        self._ranked_docs = ranked_docs

    def similarity_search(self, query, k):
        return [_FakeDoc(t, m) for t, m in self._ranked_docs[:k]]


class _FakeSparseIndex:
    def __init__(self, ranked_docs):
        self._ranked_docs = ranked_docs

    def query(self, query, k):
        return [(t, m, 1.0) for t, m in self._ranked_docs[:k]]


class _FakeReranker:
    """Scores candidates by an explicit lookup table so tests can assert
    the reranker's ordering wins over the raw fusion ordering."""

    def __init__(self, score_by_text):
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _query, text in pairs]


def test_hybrid_retriever_rerank_overrides_fusion_order():
    dense_docs = [
        ("irrelevant chunk about weather", {"source": "a", "start_index": 0}),
        ("relevant chunk about refunds", {"source": "b", "start_index": 0}),
    ]
    sparse_docs = [
        ("irrelevant chunk about weather", {"source": "a", "start_index": 0}),
    ]
    # Fusion alone would rank the weather chunk first (it's in both lists).
    reranker = _FakeReranker(
        {
            "irrelevant chunk about weather": 0.1,
            "relevant chunk about refunds": 0.9,
        }
    )

    retriever = HybridRetriever(_FakeDenseDB(dense_docs), _FakeSparseIndex(sparse_docs), reranker=reranker)
    results = retriever.retrieve("how do refunds work?", top_n=2, dense_k=2, sparse_k=1)

    assert results[0]["text"] == "relevant chunk about refunds"
    assert results[0]["score"] == 0.9


def test_hybrid_retriever_respects_top_n():
    docs = [(f"chunk {i}", {"source": "a", "start_index": i}) for i in range(5)]
    reranker = _FakeReranker({f"chunk {i}": float(i) for i in range(5)})
    retriever = HybridRetriever(_FakeDenseDB(docs), _FakeSparseIndex([]), reranker=reranker)

    results = retriever.retrieve("query", top_n=2, dense_k=5, sparse_k=0)
    assert len(results) == 2


def test_hybrid_retriever_empty_indexes_returns_empty():
    retriever = HybridRetriever(_FakeDenseDB([]), _FakeSparseIndex([]), reranker=_FakeReranker({}))
    assert retriever.retrieve("anything") == []
