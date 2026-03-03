"""
vector_store.py — Persistent FAISS dense index manager.

Responsibilities
----------------
• Build a FAISS index from child chunks (first run)
• Save the index + docstore to disk so subsequent runs skip re-embedding
• Load an existing index when the vectorstore directory is already present
• Expose a simple similarity_search_with_score wrapper used by the retriever
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.config import FAISS_INDEX_DIR
from rag.indexing.embedder import get_embedder

logger = logging.getLogger(__name__)


def build_and_save_faiss(chunks: List[Document]) -> FAISS:
    """
    Create a FAISS index from *chunks* and persist it to FAISS_INDEX_DIR.
    Called only on first run or when the index is missing / stale.
    """
    logger.info("Building FAISS index for %d chunks …", len(chunks))
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    logger.info("FAISS index saved to %s", FAISS_INDEX_DIR)
    return vectorstore


def load_faiss() -> FAISS:
    """
    Load a previously persisted FAISS index from disk.
    Raises FileNotFoundError if the index does not exist.
    """
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_INDEX_DIR}. "
            "Run indexing first (python -m src.main --index)."
        )
    logger.info("Loading FAISS index from %s …", FAISS_INDEX_DIR)
    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True,   # safe: local file we created
    )
    logger.info("FAISS index loaded (%d vectors).", vectorstore.index.ntotal)
    return vectorstore


def get_or_build_faiss(chunks: List[Document]) -> FAISS:
    """
    Load the existing index if present; otherwise build and save it.
    This is the primary entry point used by the pipeline.
    """
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if index_file.exists():
        try:
            return load_faiss()
        except Exception as exc:
            logger.warning("Failed to load existing FAISS index: %s — rebuilding.", exc)

    return build_and_save_faiss(chunks)
