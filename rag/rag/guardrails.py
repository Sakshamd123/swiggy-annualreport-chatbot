"""
guardrails.py — Anti-hallucination context validator.

Responsibilities
----------------
• Reject retrieval results whose effective relevance is below MIN_RELEVANCE_SCORE
• Surface a safe fallback response when context is insufficient
• Build the context block fed to the LLM, including source page tags
• Return PRECISE citations from top-1/2 chunks only (NOT min/max of all 6)
• Return "No relevant context found" snippet for out-of-scope queries

Changes in this revision
-------------------------
1. Citation range: uses only CITATION_TOP_K (2) top chunks, not all 6.
   Prevents "Pages 3–167" style citations.
2. Snippet: always from top-ranked chunk; overridden to
   "No relevant context found …" when fallback is triggered.
3. Confidence calibration: effective score = 0.7 × top_score +
   0.3 × query-chunk lexical overlap. Penalises generic table chunks
   returned for irrelevant queries, without weakening hallucination guards.
4. query + category are now accepted parameters so the validator can
   apply category-aware calibration.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from rag.config import MIN_RELEVANCE_SCORE, FINAL_TOP_K, CITATION_TOP_K
from rag.utils.schema import ConfidenceLevel

logger = logging.getLogger(__name__)

FALLBACK_ANSWER       = "Not found in the annual report."
NO_CONTEXT_SNIPPET    = "No relevant context found in retrieved chunks."

# Stopwords excluded from lexical overlap calculation
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "when", "where", "of", "in", "for", "by", "to", "and", "or",
    "does", "did", "do", "be", "been", "being", "its", "it", "this", "that",
    "s", "from", "as", "at", "on", "with", "their", "has", "have", "had",
    "during", "under", "per", "fiscal", "year", "fy", "india", "swiggy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_chunk_overlap(query: str, doc_content: str) -> float:
    """
    Simple unigram overlap between meaningful query tokens and top-chunk
    content.  Returns a value in [0, 1].

    This is used as a secondary relevance signal to penalise cases where
    a highly-ranked-by-embedding chunk is topically unrelated to the query
    (e.g., a generic financial table returned for a "market share" question).
    """
    query_tokens = set(re.findall(r"[a-zA-Z₹]+", query.lower())) - _STOPWORDS
    if not query_tokens:
        return 0.5  # neutral if query is all stopwords

    # Only inspect the first 3000 chars to keep this fast
    chunk_tokens = set(re.findall(r"[a-zA-Z₹]+", doc_content[:3000].lower()))
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def _compute_confidence(effective_score: float) -> ConfidenceLevel:
    """
    Map effective relevance score to High / Medium / Low.
    Thresholds tightened slightly vs original to reflect the blended score.
    """
    if effective_score >= 0.72:
        return "High"
    elif effective_score >= 0.50:
        return "Medium"
    else:
        return "Low"


def _build_citation_pages(retrieved: List[Tuple[Document, float]]) -> str:
    """
    Derive the source page range from CITATION_TOP_K (2) top chunks only.

    Before this fix the range was min(page_start)…max(page_end) across ALL
    6 retrieved chunks, producing ranges like "Pages 3–167".
    Now we anchor to the top-1 or top-2 chunks, giving tight, accurate
    citations like "Pages 46–47".
    """
    top_n   = retrieved[:CITATION_TOP_K]
    p_start = min(d.metadata.get("page_start", 0) for d, _ in top_n)
    p_end   = max(d.metadata.get("page_end",   0) for d, _ in top_n)
    return f"Pages {p_start}–{p_end}"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def validate_and_build_context(
    retrieved : List[Tuple[Document, float]],
    query     : str = "",
    category  : str = "general",
) -> Tuple[bool, str, ConfidenceLevel, str, str]:
    """
    Validate retrieved chunks and assemble the context block for the LLM.

    Parameters
    ----------
    retrieved : ranked (Document, score) list from the hybrid retriever
    query     : original user query string (used for lexical-overlap calibration)
    category  : query category ("financial" | "governance" | "general")

    Returns
    -------
    (is_valid, context_text, confidence, source_pages, context_snippet)

    is_valid        : False → caller must return FALLBACK_ANSWER immediately
    context_text    : formatted string with page-tagged chunks
    confidence      : High | Medium | Low
    source_pages    : tight citation from top-{CITATION_TOP_K} chunks only
    context_snippet : first ~300 chars of top chunk, or NO_CONTEXT_SNIPPET
    """
    if not retrieved:
        logger.warning("Guardrail triggered: no chunks retrieved.")
        return False, "", "Low", "N/A", NO_CONTEXT_SNIPPET

    top_doc, top_score = retrieved[0]

    # ── Effective relevance score ──────────────────────────────────────────
    # Blend the RRF-normalised score (top_score) with a lightweight query-
    # chunk lexical overlap to penalise generic high-scoring chunks that are
    # topically unrelated to the query.
    overlap = _query_chunk_overlap(query, top_doc.page_content)

    # For financial queries, table chunks are expected → weight overlap less
    if category == "financial":
        effective_score = 0.80 * top_score + 0.20 * overlap
    else:
        effective_score = 0.70 * top_score + 0.30 * overlap

    logger.debug(
        "Relevance calibration: top_score=%.3f overlap=%.3f "
        "effective=%.3f category=%s",
        top_score, overlap, effective_score, category,
    )

    # ── Hard threshold check ──────────────────────────────────────────────
    if effective_score < MIN_RELEVANCE_SCORE:
        logger.warning(
            "Guardrail triggered: effective_score=%.3f < threshold=%.3f "
            "(top_score=%.3f overlap=%.3f)",
            effective_score, MIN_RELEVANCE_SCORE, top_score, overlap,
        )
        return False, "", "Low", "N/A", NO_CONTEXT_SNIPPET

    confidence = _compute_confidence(effective_score)

    # ── Precise citation from top-N chunks only ───────────────────────────
    source_pages = _build_citation_pages(retrieved)

    # ── Build context block (all retrieved chunks go to LLM) ─────────────
    context_parts: List[str] = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        p_start  = doc.metadata.get("page_start", "?")
        p_end    = doc.metadata.get("page_end",   "?")
        c_type   = doc.metadata.get("chunk_type", "chunk")
        section  = doc.metadata.get("section_name", "")
        header   = (
            f"[Source {i} | Pages {p_start}–{p_end}"
            + (f" | {section}" if section else "")
            + f" | {c_type}]"
        )
        context_parts.append(f"{header}\n{doc.page_content}")

    context_text     = "\n\n---\n\n".join(context_parts)
    # Snippet: always the top chunk, trimmed to ≤300 chars
    context_snippet  = top_doc.page_content[:300].strip()

    logger.info(
        "Context validated: %d chunks | top_score=%.3f | overlap=%.3f | "
        "effective=%.3f | confidence=%s | citation=%s",
        len(retrieved), top_score, overlap, effective_score,
        confidence, source_pages,
    )
    return True, context_text, confidence, source_pages, context_snippet
