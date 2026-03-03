"""
main.py — CLI entry point for the Swiggy Annual Report RAG system.

Usage
-----
  # First run (or force-reindex):
  python -m src.main --index

  # Normal interactive QA session:
  python -m src.main

  # Single query (non-interactive):
  python -m src.main --query "What was Swiggy's net loss in FY2024?"

Startup sequence
----------------
1. Load + clean + chunk JSONL documents  (if no index exists)
2. Build or load FAISS + BM25 indices    (persistent, not rebuilt every run)
3. Construct the LangChain Runnable pipeline
4. Enter interactive CLI loop
"""

import argparse
import logging
import sys
import os

# ── Keep imports clean by adjusting sys.path for direct execution ──────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Force stdout to be utf-8 to prevent charmap errors on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Project imports ───────────────────────────────────────────────────────
from rag.config import FAISS_INDEX_DIR, BM25_INDEX_PATH, GOOGLE_API_KEY
from rag.ingestion.jsonl_loader import load_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.chunker import chunk_documents
from rag.indexing.vector_store import get_or_build_faiss, build_and_save_faiss
from rag.indexing.bm25_index import get_or_build_bm25, build_and_save_bm25
from rag.rag.rag_chain import build_rag_chain
from rag.utils.schema import RAGResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks():
    """Full ingestion pipeline → returns cleaned, chunked Documents."""
    logger.info("Loading documents from JSONL …")
    docs = load_documents()
    logger.info("Cleaning documents …")
    docs = clean_documents(docs)
    logger.info("Chunking documents …")
    chunks = chunk_documents(docs)
    logger.info("Ingestion complete: %d chunks ready.", len(chunks))
    return chunks


def _ensure_indices(force_rebuild: bool = False):
    """
    Load or build FAISS + BM25 indices.
    Returns (faiss_store, bm25_index, bm25_chunks).
    """
    index_missing = not (FAISS_INDEX_DIR / "index.faiss").exists() or \
                    not BM25_INDEX_PATH.exists()

    if force_rebuild or index_missing:
        if force_rebuild:
            logger.info("Force-rebuilding indices …")
        else:
            logger.info("No existing index found — building for the first time …")
            logger.info("This will download ~1.3 GB embedding model and may take "
                        "several minutes on CPU. Subsequent runs will be fast.")

        chunks = _load_chunks()
        faiss_store          = build_and_save_faiss(chunks)
        bm25_index, bm25_chunks = build_and_save_bm25(chunks)
    else:
        # Lazy load: just read from disk
        chunks = _load_chunks()
        faiss_store             = get_or_build_faiss(chunks)
        bm25_index, bm25_chunks = get_or_build_bm25(chunks)

    return faiss_store, bm25_index, bm25_chunks


def _print_response(response: RAGResponse):
    """Pretty-print a RAGResponse to the terminal."""
    print()
    print("=" * 70)
    print("  ANSWER")
    print("=" * 70)
    print(response.answer)
    print()

    print("-" * 70)
    print(f"  Source     : {response.source_pages}")
    print(f"  Confidence : {response.confidence}")
    print(f"  Category   : {response.query_category.upper()}")
    print("-" * 70)

    # Always show snippet — for valid answers this is the top-chunk excerpt;
    # for fallback/"Not found" answers it shows "No relevant context found..."
    snippet = response.context_snippet
    if snippet:
        print("  Supporting Context:")
        print()
        for line in snippet.split("\n"):
            print(f"    {line}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Swiggy Annual Report RAG System (Backend CLI)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Force rebuild FAISS + BM25 indices from scratch.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single non-interactive query and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Environment check ──────────────────────────────────────────────────
    if not GOOGLE_API_KEY:
        logger.error(
            "GOOGLE_API_KEY is not set.\n"
            "Create a .env file in the project root:\n"
            "  GOOGLE_API_KEY=your_key_here"
        )
        sys.exit(1)

    # ── Index loading / building ───────────────────────────────────────────
    try:
        faiss_store, bm25_index, bm25_chunks = _ensure_indices(
            force_rebuild=args.index
        )
    except Exception as exc:
        logger.error("Failed to initialise indices: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Build pipeline ─────────────────────────────────────────────────────
    logger.info("Initialising RAG pipeline …")
    rag_pipeline = build_rag_chain(faiss_store, bm25_index, bm25_chunks)
    logger.info("Pipeline ready.\n")

    # ── Single-query mode ──────────────────────────────────────────────────
    if args.query:
        response: RAGResponse = rag_pipeline.invoke(args.query)
        _print_response(response)
        return

    # ── Interactive mode ───────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    Swiggy Annual Report 2023-2024 — RAG Question Answering      ║")
    print("║    Type 'exit' or 'quit' to stop.                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            query = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            response: RAGResponse = rag_pipeline.invoke(query)
            _print_response(response)
        except Exception as exc:
            logger.error("Error processing query: %s", exc, exc_info=args.debug)
            print(f"\n[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
