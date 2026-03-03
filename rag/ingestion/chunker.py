"""
chunker.py — Content-aware Parent-Child chunking for the Swiggy RAG pipeline.

Strategy
--------
Parent  = one JSONL record (already loaded as a LangChain Document)
Children = 800–1000 token sub-chunks with 150-token overlap

Each child chunk:
  • inherits page_start / page_end / document_id / source from its parent
  • is tagged as "table_chunk" or "text_chunk" based on content heuristics
  • page boundaries are re-estimated proportionally when a parent spans pages

The RecursiveCharacterTextSplitter is configured so markdown table rows
(containing |) are treated as hard split boundaries — tables are NEVER
split mid-row.
"""

import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TABLE_KEYWORDS,
    TABLE_PIPE_RATIO,
)

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for BGE embeddings on English text
_CHARS_PER_TOKEN = 4

_CHUNK_SIZE_CHARS    = CHUNK_SIZE    * _CHARS_PER_TOKEN   # ~3600 chars
_CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * _CHARS_PER_TOKEN   # ~600  chars

# Splitter — try to break on double-newlines first, then single, then space
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE_CHARS,
    chunk_overlap=_CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    keep_separator=True,
)

_TABLE_KW_LOWER = [kw.lower() for kw in TABLE_KEYWORDS]


def _is_table_chunk(text: str) -> bool:
    """
    Heuristic: a chunk is considered table-heavy if
      (a) the ratio of pipe characters (|) exceeds TABLE_PIPE_RATIO, OR
      (b) it contains a financial keyword.
    """
    if len(text) == 0:
        return False
    pipe_ratio = text.count("|") / len(text)
    if pipe_ratio >= TABLE_PIPE_RATIO:
        return True
    text_lower = text.lower()
    return any(kw in text_lower for kw in _TABLE_KW_LOWER)


def _estimate_page_range(
    parent_start: int,
    parent_end: int,
    char_offset: int,
    parent_len: int,
) -> tuple[int, int]:
    """
    Linearly interpolate the page range a chunk belongs to inside its parent.
    Returns (chunk_page_start, chunk_page_end).
    """
    total_pages = max(parent_end - parent_start, 1)
    fraction    = char_offset / max(parent_len, 1)
    estimated   = parent_start + int(fraction * total_pages)
    # Give the chunk a 1-page window so citations are tight but not wrong
    chunk_start = max(parent_start, estimated)
    chunk_end   = min(parent_end,   estimated + 1)
    return chunk_start, chunk_end


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split each parent document into child chunks.

    Returns a flat list of child Documents — each with full metadata
    required for citation grounding.
    """
    all_chunks: List[Document] = []

    for parent in documents:
        p_start  = parent.metadata["page_start"]
        p_end    = parent.metadata["page_end"]
        doc_id   = parent.metadata["document_id"]
        source   = parent.metadata["source"]
        c_type   = parent.metadata["content_type"]
        text     = parent.page_content
        text_len = len(text)

        # Split the parent text into raw text pieces
        raw_chunks = _SPLITTER.split_text(text)

        # Track approximate character offset to estimate page range
        char_cursor = 0

        for raw in raw_chunks:
            if not raw.strip():
                continue

            chunk_type = "table_chunk" if _is_table_chunk(raw) else "text_chunk"
            c_start, c_end = _estimate_page_range(
                p_start, p_end, char_cursor, text_len
            )

            chunk_doc = Document(
                page_content=raw,
                metadata={
                    "document_id":  doc_id,
                    "page_start":   c_start,
                    "page_end":     c_end,
                    "content_type": c_type,
                    "chunk_type":   chunk_type,
                    "source":       source,
                    "char_length":  len(raw),
                    # Inherit section label from parent (added in jsonl_loader)
                    "section_name": parent.metadata.get("section_name", ""),
                },
            )
            all_chunks.append(chunk_doc)
            char_cursor += len(raw) - _CHUNK_OVERLAP_CHARS
            char_cursor = max(char_cursor, 0)

    logger.info(
        "Chunked %d parent docs → %d child chunks.",
        len(documents),
        len(all_chunks),
    )
    if all_chunks:
        table_count = sum(
            1 for c in all_chunks if c.metadata["chunk_type"] == "table_chunk"
        )
        logger.info(
            "  table_chunks: %d  |  text_chunks: %d",
            table_count,
            len(all_chunks) - table_count,
        )

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from rag.ingestion.jsonl_loader import load_documents
    from rag.ingestion.cleaner import clean_documents

    docs   = clean_documents(load_documents())
    chunks = chunk_documents(docs)
    for c in chunks[:5]:
        print(
            f"[{c.metadata['chunk_type']}] "
            f"p{c.metadata['page_start']}–{c.metadata['page_end']} | "
            f"{len(c.page_content)} chars"
        )
        print(c.page_content[:120])
        print("---")
