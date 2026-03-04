"""
parent_context.py — Parent Document Retrieval: expand small retrieved chunks
                    into their surrounding section context.

Design
------
Each retrieved chunk carries metadata:
    page_start, page_end, section_name, document_id

"Parent context" for a chunk = all chunks from the same *section* whose
page range overlaps or is adjacent to the retrieved chunk.

This gives the LLM a broader view (e.g. the full P&L table section rather
than a single row), improving financial table understanding.

Constraints
-----------
• MAX_PARENT_CHARS : hard cap on total expanded context length
• MAX_CONTEXTS     : at most 6 logical context blocks passed to the LLM
• No duplicate sections — if two retrieved chunks expand into the same
  section, the expanded block is included only once.
• Scores are preserved from the retrieved chunks (not diluted by expansion).
• The RRF scores, guardrail validation, and post-processing are ALL
  untouched — this step only replaces chunk.page_content with a wider
  excerpt before the context is assembled for the LLM.

Integration point
-----------------
Called inside step_validate() in rag_chain.py, BEFORE
validate_and_build_context(), so guardrails operate on the expanded content.
"""

import hashlib
import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hard cap: total characters across all expanded contexts fed to LLM.
# ~4000 tokens × 4 chars/token = 16 000 chars.  We stay well under that.
MAX_PARENT_CHARS = 12_000

# Max number of independent context blocks passed forward (mirrors FINAL_TOP_K)
MAX_CONTEXTS = 6

# How many extra chunks to include on each side of a retrieved chunk
# within the same section (page-adjacency window).
_PAGE_WINDOW = 2        # pages on each side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_key(doc: Document) -> str:
    """Unique key for a section block = section_name + document_id."""
    section = doc.metadata.get("section_name", "unknown")
    doc_id  = doc.metadata.get("document_id",  "unknown")
    return f"{section}::{doc_id}"


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _pages_overlap_or_adjacent(
    a_start: int, a_end: int,
    b_start: int, b_end: int,
    window: int = _PAGE_WINDOW,
) -> bool:
    """
    True if chunk B is within *window* pages of chunk A.
    Used to build a tight "parent section" from the corpus.
    """
    return (
        a_start - window <= b_end and
        b_start <= a_end + window
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def expand_to_parent_context(
    retrieved: List[Tuple[Document, float]],
    all_chunks: List[Document],
) -> List[Tuple[Document, float]]:
    """
    Expand each retrieved chunk into its surrounding section context.

    Parameters
    ----------
    retrieved   : ranked (Document, score) list from hybrid retrieval + rerank
    all_chunks  : the full BM25 chunk corpus (used as the expansion pool)

    Returns
    -------
    A new ranked list of (Document, score) pairs where each Document's
    page_content has been expanded to include surrounding section chunks.
    The list has at most MAX_CONTEXTS entries and respects MAX_PARENT_CHARS.

    The original retrieval scores are preserved unchanged.
    """
    if not retrieved or not all_chunks:
        return retrieved

    expanded_results: List[Tuple[Document, float]] = []
    seen_section_keys: set = set()
    total_chars = 0

    for chunk, score in retrieved:
        if len(expanded_results) >= MAX_CONTEXTS:
            break

        sec_key = _section_key(chunk)

        # If this section was already expanded by a higher-ranked chunk, skip
        if sec_key in seen_section_keys:
            logger.debug(
                "Parent expansion: skipping duplicate section %r", sec_key
            )
            continue

        c_start = chunk.metadata.get("page_start", 0)
        c_end   = chunk.metadata.get("page_end",   0)
        c_sec   = chunk.metadata.get("section_name", "")
        c_docid = chunk.metadata.get("document_id", "")

        # Collect all corpus chunks in the same section and page window
        sibling_chunks = []
        for candidate in all_chunks:
            if candidate.metadata.get("section_name", "") != c_sec:
                continue
            if candidate.metadata.get("document_id",  "") != c_docid:
                continue
            b_start = candidate.metadata.get("page_start", 0)
            b_end   = candidate.metadata.get("page_end",   0)
            if _pages_overlap_or_adjacent(c_start, c_end, b_start, b_end):
                sibling_chunks.append(candidate)

        if not sibling_chunks:
            # No siblings found — use the chunk as-is
            sibling_chunks = [chunk]

        # Sort siblings by page order for coherent reading
        sibling_chunks.sort(
            key=lambda d: (d.metadata.get("page_start", 0),
                           d.metadata.get("page_end",   0))
        )

        # Deduplicate siblings by content hash
        seen_hashes: set = set()
        unique_siblings = []
        for s in sibling_chunks:
            h = _content_hash(s.page_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_siblings.append(s)

        # Build merged content, respecting total char budget
        merged_parts = []
        for s in unique_siblings:
            part_len = len(s.page_content)
            if total_chars + part_len > MAX_PARENT_CHARS:
                logger.debug(
                    "Parent expansion: char budget reached at section %r",
                    sec_key,
                )
                break
            merged_parts.append(s.page_content)
            total_chars += part_len

        if not merged_parts:
            # Budget exhausted even before first chunk — add the original chunk
            merged_parts = [chunk.page_content]

        merged_text = "\n\n".join(merged_parts)

        # Build an expanded Document — metadata comes from the original
        # retrieved chunk (preserves page_start/page_end for citation).
        # We widen page_end to the last sibling included for accuracy.
        last_included = unique_siblings[len(merged_parts) - 1]
        expanded_metadata = dict(chunk.metadata)
        expanded_metadata["page_end"] = max(
            chunk.metadata.get("page_end", 0),
            last_included.metadata.get("page_end", 0),
        )
        expanded_metadata["parent_expanded"] = True
        expanded_metadata["siblings_included"] = len(merged_parts)

        expanded_doc = Document(
            page_content=merged_text,
            metadata=expanded_metadata,
        )

        expanded_results.append((expanded_doc, score))
        seen_section_keys.add(sec_key)

        logger.debug(
            "Parent expansion: section=%r | original_pages=%d–%d | "
            "siblings=%d | merged_chars=%d",
            sec_key, c_start, c_end,
            len(merged_parts), len(merged_text),
        )

    logger.info(
        "Parent context expansion: %d retrieved → %d contexts | "
        "total_chars=%d",
        len(retrieved), len(expanded_results), total_chars,
    )
    return expanded_results
