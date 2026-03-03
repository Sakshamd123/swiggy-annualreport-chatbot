"""
embedder.py — Singleton BGE embedding wrapper for FAISS indexing + query time.

Model: BAAI/bge-large-en-v1.5
  • Normalised L2 embeddings (cosine-compatible)
  • CPU-friendly
  • Batch support to bound peak memory usage
"""

import logging
from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Return a cached BGE embedding model instance.
    The model is downloaded once and reused for both indexing and retrieval.
    This avoids the ~1 GB model reload penalty on every query.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,   # unit vectors → cosine sim via dot product
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )
    logger.info("Embedding model loaded.")
    return embedder
