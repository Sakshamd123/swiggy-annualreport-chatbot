"""
reranker.py — Optional, modular reranking stage.

Design goals
------------
• Must not break the pipeline if disabled (ENABLE_RERANKER = False)
• Default implementation: score-weighted deduplication (lightweight)
• Hook for cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)

To enable a cross-encoder reranker in future:
  1. pip install sentence-transformers
  2. Uncomment the CrossEncoderReranker section below
  3. Set ENABLE_RERANKER = True in config.py
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Set to True to activate reranking (currently uses similarity-based rerank)
ENABLE_RERANKER = False


def rerank(
    query: str,
    results: List[Tuple[Document, float]],
    top_k: int = 6,
) -> List[Tuple[Document, float]]:
    """
    Optional rerank stage.

    If ENABLE_RERANKER is False (default), pass results straight through.

    If True, apply a lightweight similarity-based rerank:
      • Reward chunks whose content has higher lexical overlap with the query
        (simple unigram overlap ratio as a proxy for cross-encoder scores)
      • Combined score = 0.7 × original + 0.3 × lexical_overlap
    This keeps the module non-breaking and low-latency on CPU.
    """
    if not ENABLE_RERANKER:
        logger.debug("Reranker disabled — passing %d results through.", len(results))
        return results[:top_k]

    logger.debug("Reranking %d results …", len(results))
    query_tokens = set(query.lower().split())

    reranked: List[Tuple[Document, float]] = []
    for doc, score in results:
        chunk_tokens = set(doc.page_content.lower().split())
        if query_tokens:
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        else:
            overlap = 0.0
        combined = 0.7 * score + 0.3 * overlap
        reranked.append((doc, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]

    # ── Future cross-encoder hook ──────────────────────────────────────────
    # from sentence_transformers import CrossEncoder
    # _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    #
    # pairs = [(query, doc.page_content) for doc, _ in results]
    # ce_scores = _cross_encoder.predict(pairs)
    # reranked = sorted(
    #     zip([d for d, _ in results], ce_scores),
    #     key=lambda x: x[1], reverse=True
    # )
    # return [(doc, float(s)) for doc, s in reranked[:top_k]]
