"""
hybrid_retriever.py — Merge FAISS dense + BM25 sparse results.

Algorithm
---------
1. Run FAISS similarity search → top FAISS_TOP_K (doc, score) pairs
2. Run BM25 keyword search    → top BM25_TOP_K   (doc, score) pairs
3. Reciprocal Rank Fusion     → blend both ranked lists
4. Deduplicate by content hash
5. Optionally boost table_chunks for financial queries
6. Return final top FINAL_TOP_K chunks sorted by fused score

Reciprocal Rank Fusion (RRF) is used instead of simple score averaging
because FAISS cosine scores and BM25 scores live on different scales.
RRF is rank-based and naturally fuses them without normalisation bias.
"""

import hashlib
import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.config import FAISS_TOP_K, BM25_TOP_K, FINAL_TOP_K
from rag.indexing.bm25_index import bm25_search

logger = logging.getLogger(__name__)

# RRF constant — 60 is the standard value from the original paper
_RRF_K = 60

# Boost factor applied to table_chunks for financial queries
_TABLE_BOOST = 1.15


def _content_hash(text: str) -> str:
    """Short hash for deduplication (first 12 hex chars of MD5)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _rrf_score(rank: int) -> float:
    """RRF score for a document appearing at *rank* (1-indexed)."""
    return 1.0 / (_RRF_K + rank)


def hybrid_retrieve(
    query: str,
    faiss_store: FAISS,
    bm25_index: BM25Okapi,
    bm25_chunks: List[Document],
    boost_table_chunks: bool = False,
) -> List[Tuple[Document, float]]:
    """
    Perform hybrid retrieval and return a deduplicated list of
    (Document, fused_score) pairs, highest score first.

    Parameters
    ----------
    query              : raw user query string
    faiss_store        : loaded FAISS vectorstore
    bm25_index         : loaded BM25Okapi instance
    bm25_chunks        : corpus Documents aligned with bm25_index
    boost_table_chunks : if True, table_chunk scores get a slight boost
                         (used for financial queries)
    """
    # ── Dense retrieval ────────────────────────────────────────────────────
    faiss_results: List[Tuple[Document, float]] = (
        faiss_store.similarity_search_with_score(query, k=FAISS_TOP_K)
    )
    # FAISS returns (doc, L2/cosine distance); lower = better for L2,
    # but we saved normalised embeddings so it uses inner-product ≈ cosine.
    # LangChain FAISS actually returns the cosine-distance (1-sim), so
    # we convert to similarity: sim = 1 - dist  (clamp to [0,1])
    faiss_ranked: List[Tuple[str, Document, float]] = []
    for rank, (doc, dist) in enumerate(faiss_results, start=1):
        sim   = max(0.0, 1.0 - dist)   # convert distance → similarity
        score = _rrf_score(rank)
        faiss_ranked.append((_content_hash(doc.page_content), doc, score))

    # ── Sparse retrieval ───────────────────────────────────────────────────
    bm25_results = bm25_search(bm25_index, bm25_chunks, query, top_k=BM25_TOP_K)
    bm25_ranked: List[Tuple[str, Document, float]] = []
    for rank, (doc, _norm_score) in enumerate(bm25_results, start=1):
        score = _rrf_score(rank)
        bm25_ranked.append((_content_hash(doc.page_content), doc, score))

    # ── RRF fusion ─────────────────────────────────────────────────────────
    fused: dict[str, Tuple[Document, float]] = {}

    for h, doc, score in faiss_ranked:
        if h not in fused:
            fused[h] = (doc, 0.0)
        c_doc, c_score = fused[h]
        fused[h] = (c_doc, c_score + score)

    for h, doc, score in bm25_ranked:
        if h not in fused:
            fused[h] = (doc, 0.0)
        c_doc, c_score = fused[h]
        fused[h] = (c_doc, c_score + score)

    # ── Optional table boost ───────────────────────────────────────────────
    if boost_table_chunks:
        boosted: dict[str, Tuple[Document, float]] = {}
        for h, (doc, score) in fused.items():
            if doc.metadata.get("chunk_type") == "table_chunk":
                score *= _TABLE_BOOST
            boosted[h] = (doc, score)
        fused = boosted

    # ── Sort + trim + Normalize ────────────────────────────────────────────
    # Max theoretical RRF score without boost is ~ 0.0327 (rank 1 in both FAISS and BM25)
    # Normalize by 0.033 so scores comfortably map to [0.0, 1.0] for the guardrails.
    sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    final = [(doc, min(score / 0.033, 1.0)) for doc, score in sorted_results[:FINAL_TOP_K]]

    logger.debug(
        "Hybrid retrieval: FAISS=%d, BM25=%d → merged=%d → final=%d",
        len(faiss_ranked),
        len(bm25_ranked),
        len(fused),
        len(final),
    )

    return final
