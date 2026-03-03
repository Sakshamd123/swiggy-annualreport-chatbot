"""
config.py — Central configuration for the Swiggy RAG backend.
All tunable parameters live here; no magic constants elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from rag/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = _ROOT / "data"
JSONL_PATH      = DATA_DIR / "swiggy_multimodal.jsonl"
VECTORSTORE_DIR = _ROOT / "vectorstore"
FAISS_INDEX_DIR = VECTORSTORE_DIR / "faiss_index"
BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_index.pkl"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE  = 16        # conservative for CPU
EMBEDDING_NORMALIZE   = True      # cosine-compatible embeddings

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE        = 500           # reduced 900→500 for finer line-item retrieval
CHUNK_OVERLAP     = 150           # token overlap (30% of 500 — needed for this doc's structure)
# Heuristics for detecting table-heavy chunks
TABLE_KEYWORDS    = ["₹", "INR", "EBITDA", "revenue", "loss", "profit",
                     "crore", "lakh", "million", "balance sheet",
                     "cash flow", "earnings", "net loss", "gross", "%"]
TABLE_PIPE_RATIO  = 0.03          # fraction of '|' chars to flag as table chunk

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
FAISS_TOP_K  = 5      # semantic neighbours from FAISS
BM25_TOP_K   = 5      # keyword hits from BM25
FINAL_TOP_K  = 6      # contexts fed to LLM after merge + dedup

# Guardrail: minimum relevance score (combines RRF + lexical overlap)
# Raised from 0.30 → 0.42 to reduce false-positive "High confidence" for
# out-of-scope queries that happen to surface high-scoring generic chunks.
MIN_RELEVANCE_SCORE = 0.42

# Citation: number of top-ranked chunks used to derive source page range.
# Using only top-2 prevents "Pages 3–167" style citations.
CITATION_TOP_K = 2

# ---------------------------------------------------------------------------
# LLM — Gemini 2.5 Flash
# ---------------------------------------------------------------------------
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_TEMP     = 0               # fully deterministic

# ---------------------------------------------------------------------------
# Source label (shown in citations)
# ---------------------------------------------------------------------------
SOURCE_LABEL = "Swiggy Annual Report 2023-2024"
