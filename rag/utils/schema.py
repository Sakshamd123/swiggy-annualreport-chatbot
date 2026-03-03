"""
schema.py — Shared data models for the Swiggy RAG pipeline.
Provides typed, validated data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


# ---------------------------------------------------------------------------
# Chunk / Document schema
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Metadata that every chunk MUST carry to enable citation grounding."""
    document_id: str
    page_start: int
    page_end: int
    content_type: str          # "structured_text", etc.
    chunk_type: str            # "table_chunk" | "text_chunk"
    source: str                # canonical source label
    char_length: int = 0


# ---------------------------------------------------------------------------
# Query classification schema
# ---------------------------------------------------------------------------

QueryCategory = Literal["financial", "governance", "general"]


@dataclass
class ClassifiedQuery:
    """Output of the lightweight query classifier."""
    query: str
    category: QueryCategory
    boost_table_chunks: bool   # True for financial queries


# ---------------------------------------------------------------------------
# Retrieval result schema
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its score and provenance."""
    content: str
    metadata: ChunkMetadata
    score: float              # higher = more relevant


# ---------------------------------------------------------------------------
# RAG pipeline output schema
# ---------------------------------------------------------------------------

ConfidenceLevel = Literal["High", "Medium", "Low"]


@dataclass
class RAGResponse:
    """Final structured output from the full RAG pipeline."""
    answer: str
    source_pages: str                           # e.g. "Pages 43–58"
    confidence: ConfidenceLevel
    context_snippet: str                        # short excerpt shown to user
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    query_category: QueryCategory = "general"
    fallback: bool = False                      # True when "Not found" is returned
