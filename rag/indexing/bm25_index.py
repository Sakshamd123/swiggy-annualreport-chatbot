"""
bm25_index.py — Sparse BM25 keyword index (rank-bm25).

Responsibilities
----------------
• Tokenise chunk corpus (simple whitespace + lower-case tokeniser)
• Build BM25Okapi index
• Persist to disk with pickle (tiny file, fast I/O)
• Expose top-k retrieval aligned with FAISS results

BM25 excels at exact-match retrieval for financial terms like
"EBITDA", "IPO", "net loss" that dense embeddings sometimes miss.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from rag.config import BM25_INDEX_PATH

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9₹%\.]+")


def _tokenize(text: str) -> List[str]:
    """
    Lower-case word tokeniser that retains digits, currency, and percent
    signs — important for financial term matching.
    """
    return _TOKEN_RE.findall(text.lower())


def build_and_save_bm25(chunks: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """
    Build a BM25 index from *chunks* and persist it to BM25_INDEX_PATH.
    Returns (bm25_index, chunks) so callers can reference the same list.
    """
    logger.info("Building BM25 index for %d chunks …", len(chunks))
    corpus_tokens = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh)

    logger.info("BM25 index saved to %s", BM25_INDEX_PATH)
    return bm25, chunks


def load_bm25() -> Tuple[BM25Okapi, List[Document]]:
    """
    Load the persisted BM25 index + corpus from disk.
    Raises FileNotFoundError if not present.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run indexing first."
        )
    logger.info("Loading BM25 index from %s …", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as fh:
        payload = pickle.load(fh)
    bm25   = payload["bm25"]
    chunks = payload["chunks"]
    logger.info("BM25 index loaded (%d documents).", len(chunks))
    return bm25, chunks


def get_or_build_bm25(
    chunks: List[Document],
) -> Tuple[BM25Okapi, List[Document]]:
    """
    Load existing BM25 index if present; otherwise build and save it.
    """
    if BM25_INDEX_PATH.exists():
        try:
            return load_bm25()
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s — rebuilding.", exc)

    return build_and_save_bm25(chunks)


def bm25_search(
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Return the top-k BM25 matches as (Document, normalised_score) pairs.
    Scores are normalised to [0, 1] against the maximum score in the result.
    """
    query_tokens = _tokenize(query)
    scores       = bm25.get_scores(query_tokens)

    # Pair each chunk with its score and sort descending
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top    = scored[:top_k]

    max_score = top[0][1] if top and top[0][1] > 0 else 1.0

    results: List[Tuple[Document, float]] = []
    for idx, raw_score in top:
        normalised = raw_score / max_score if max_score > 0 else 0.0
        results.append((chunks[idx], normalised))

    return results
