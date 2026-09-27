"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with Reciprocal
Rank Fusion, then cross-encoder reranked. Replaces the original's bare
`db.as_retriever(search_kwargs={'k': 2})` dense-only lookup.

Dense embeddings are great at paraphrase/semantic match but miss exact
terms, codes, and rare names; BM25 is the reverse. Fusing both and then
reranking with a cross-encoder (which scores the query and passage
jointly, rather than via independent vectors) is the standard
production-grade retrieval recipe as of 2025-2026.
"""
from app import config


class _FastEmbedRerankerAdapter:
    """Wraps fastembed's ONNX-runtime TextCrossEncoder behind the same
    .predict(pairs) shape sentence_transformers.CrossEncoder exposes, so
    HybridRetriever.retrieve() doesn't need to know which backend is
    loaded. Chosen over sentence-transformers specifically to avoid
    pulling in a ~600MB PyTorch install for a task this small -- fastembed
    covers the exact same reranker model (as an ONNX export) in ~1/4 the
    footprint.
    """

    def __init__(self, model_name):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def predict(self, pairs):
        # HybridRetriever always builds pairs against a single query, so
        # every pair shares the same [0] element.
        query = pairs[0][0]
        documents = [doc for _query, doc in pairs]
        return list(self._model.rerank(query, documents))


def _candidate_id(metadata, text):
    source = metadata.get("source")
    start_index = metadata.get("start_index")
    if source is not None and start_index is not None:
        return (source, start_index)
    return hash(text)


def reciprocal_rank_fusion(ranked_lists, k=None):
    """ranked_lists: list of lists of (text, metadata), each already
    sorted best-first. Returns fused list of (text, metadata, rrf_score)
    sorted best-first, deduped across lists."""
    k = k if k is not None else config.RRF_K
    scores = {}
    payload = {}

    for ranked in ranked_lists:
        for rank, (text, metadata) in enumerate(ranked):
            cid = _candidate_id(metadata, text)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payload.setdefault(cid, (text, metadata))

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(payload[cid][0], payload[cid][1], score) for cid, score in fused]


class HybridRetriever:
    def __init__(self, dense_db, sparse_index, reranker=None):
        self.dense_db = dense_db
        self.sparse_index = sparse_index
        self._reranker = reranker  # lazily loaded CrossEncoder if None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = _FastEmbedRerankerAdapter(config.RERANKER_MODEL)
        return self._reranker

    def _dense_candidates(self, query, k):
        results = self.dense_db.similarity_search(query, k=k)
        return [(doc.page_content, doc.metadata) for doc in results]

    def _sparse_candidates(self, query, k):
        results = self.sparse_index.query(query, k=k)
        return [(text, metadata) for text, metadata, _score in results]

    def retrieve(self, query, top_n=None, dense_k=None, sparse_k=None):
        top_n = top_n if top_n is not None else config.RERANK_TOP_N
        dense_k = dense_k if dense_k is not None else config.DENSE_TOP_K
        sparse_k = sparse_k if sparse_k is not None else config.SPARSE_TOP_K

        dense = self._dense_candidates(query, dense_k)
        sparse = self._sparse_candidates(query, sparse_k)
        fused = reciprocal_rank_fusion([dense, sparse])

        if not fused:
            return []

        pairs = [(query, text) for text, _metadata, _score in fused]
        if hasattr(self.reranker, "predict"):
            rerank_scores = self.reranker.predict(pairs)
        else:
            # test doubles may just be callables
            rerank_scores = self.reranker(pairs)

        reranked = sorted(
            zip(fused, rerank_scores), key=lambda item: item[1], reverse=True
        )
        top = reranked[:top_n]
        return [
            {"text": text, "metadata": metadata, "score": float(score)}
            for (text, metadata, _rrf_score), score in top
        ]
