"""
jsonl_loader.py — Load and convert swiggy_multimodal.jsonl into
                   LangChain Document objects with rich metadata.

Each JSONL record maps to exactly ONE LangChain Document (the "parent").
The chunker will break these parents into child chunks later.

Loading strategy (two-path)
----------------------------
PRIMARY:   Load from swiggy_multimodal.jsonl  (fast, structured)
FALLBACK:  If JSONL missing, find any PDF in data/, extract text with
           pypdf, group into ~15-page records, save as JSONL for future
           runs, then return the same Document list.

This satisfies the assignment requirement of "Load the Swiggy Annual
Report PDF" while keeping the fast JSONL path intact.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from rag.config import DATA_DIR, JSONL_PATH, SOURCE_LABEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# How many PDF pages to bundle into one parent Document (mirrors the original
# JSONL structure of ~15 pages per record).
# ---------------------------------------------------------------------------
_PDF_PAGES_PER_BATCH = 15

# ---------------------------------------------------------------------------
# Section mapping (Swiggy Annual Report FY 2023-24 page layout)
# Derived from the official PDF table-of-contents.
# ---------------------------------------------------------------------------
_SECTION_MAP = [
    (1,   2,   "Corporate Information"),
    (3,   15,  "Board's Report"),
    (16,  30,  "Board's Report (continued)"),
    (31,  42,  "Independent Auditor's Report (Standalone)"),
    (43,  60,  "Standalone Financial Statements"),
    (61,  90,  "Notes to Standalone Financial Statements"),
    (91,  107, "Notes to Standalone FS (continued)"),
    (108, 120, "Independent Auditor's Report (Consolidated)"),
    (121, 137, "Consolidated Financial Statements"),
    (138, 155, "Notes to Consolidated Financial Statements"),
    (156, 167, "Notes to Consolidated FS (continued)"),
]


def _section_name(page_start: int) -> str:
    """Return the document section that *starts at or contains* page_start."""
    for lo, hi, name in _SECTION_MAP:
        if lo <= page_start <= hi:
            return name
    return "Annexures"


# ---------------------------------------------------------------------------
# PDF fallback helpers
# ---------------------------------------------------------------------------

def _find_pdf(search_dir: Path) -> Optional[Path]:
    """
    Locate the annual report PDF in the data directory.
    Preference order:
      1. swiggy_annual_report.pdf  (canonical name used in docs)
      2. Any PDF whose name contains "annual" (case-insensitive)
      3. First PDF found in the directory
    """
    candidates = list(search_dir.glob("*.pdf"))
    if not candidates:
        return None

    for c in candidates:
        if c.name.lower() == "swiggy_annual_report.pdf":
            return c
    for c in candidates:
        if "annual" in c.name.lower():
            return c
    return candidates[0]


def _pdf_to_documents(pdf_path: Path) -> List[Document]:
    """
    Extract text from a PDF using pypdf and group pages into
    _PDF_PAGES_PER_BATCH-page LangChain Documents.

    Returns a list of Documents with the same metadata schema as the
    JSONL primary path, so the rest of the pipeline is unaffected.
    """
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF fallback loading.\n"
            "Install it with:  pip install pypdf>=3.0.0"
        ) from exc

    logger.info("Extracting text from PDF: %s", pdf_path.name)
    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    logger.info("PDF has %d pages. Grouping into %d-page batches …",
                total_pages, _PDF_PAGES_PER_BATCH)

    documents: List[Document] = []

    for batch_idx, batch_start in enumerate(
        range(0, total_pages, _PDF_PAGES_PER_BATCH)
    ):
        batch_end_idx = min(batch_start + _PDF_PAGES_PER_BATCH - 1,
                            total_pages - 1)
        # Extract text, prepend page header so chunker context is clear
        page_texts: List[str] = []
        for page_idx in range(batch_start, batch_end_idx + 1):
            raw = reader.pages[page_idx].extract_text() or ""
            raw = raw.strip()
            if raw:
                page_texts.append(f"## Page {page_idx + 1}\n{raw}")

        content = "\n\n".join(page_texts).strip()
        if not content:
            logger.debug("Batch %d (pages %d–%d) is empty — skipping.",
                         batch_idx, batch_start + 1, batch_end_idx + 1)
            continue

        page_s = batch_start + 1
        page_e = batch_end_idx + 1

        doc = Document(
            page_content=content,
            metadata={
                "document_id":  f"pdf_batch_{page_s}_{page_e}",
                "page_start":   page_s,
                "page_end":     page_e,
                "content_type": "structured_text",
                "source":       SOURCE_LABEL,
                "section_name": _section_name(page_s),
                "char_length":  len(content),
                # Detect table-like content by pipe character density
                "has_tables":   (content.count("|") / max(len(content), 1)) > 0.02,
            },
        )
        documents.append(doc)

    logger.info("PDF extraction complete: %d parent documents produced.",
                len(documents))
    return documents


def _save_as_jsonl(documents: List[Document], output_path: Path) -> None:
    """
    Save a list of LangChain Documents as a JSONL file compatible with the
    load_documents() reader.  Called once after PDF extraction so subsequent
    runs use the faster JSONL path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for doc in documents:
            m = doc.metadata
            record = {
                "document_id":  m.get("document_id", "unknown"),
                "page_start":   m.get("page_start", 0),
                "page_end":     m.get("page_end", 0),
                "content_type": m.get("content_type", "structured_text"),
                "content":      doc.page_content,
                "metadata": {
                    "char_length": m.get("char_length", len(doc.page_content)),
                    "has_tables":  m.get("has_tables", False),
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d documents as JSONL → %s", len(documents), output_path)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_documents(jsonl_path: Path = JSONL_PATH) -> List[Document]:
    """
    Return parent LangChain Documents ready for chunking.

    PRIMARY PATH  (normal):
        Reads swiggy_multimodal.jsonl.

    FALLBACK PATH (if JSONL missing):
        Searches data/ for a PDF, extracts text with pypdf,
        saves result as swiggy_multimodal.jsonl for future runs,
        then returns the same Document list.

    Either way, the returned Documents have identical schema so the
    downstream pipeline (cleaner → chunker → indexer) is unaffected.
    """
    # ── FALLBACK: JSONL missing → try PDF ─────────────────────────────────
    if not jsonl_path.exists():
        logger.warning(
            "JSONL not found at: %s\nSearching for a PDF in: %s",
            jsonl_path, jsonl_path.parent,
        )
        pdf_path = _find_pdf(jsonl_path.parent)
        if pdf_path is None:
            raise FileNotFoundError(
                f"Neither '{jsonl_path.name}' nor any PDF found in '{jsonl_path.parent}'.\n"
                "Place the Swiggy Annual Report PDF in the data/ directory.\n"
                "Download: https://www.bseindia.com/bseplus/AnnualReport/"
                "544225/10424544225.pdf\n"
                "Rename to: swiggy_annual_report.pdf"
            )

        documents = _pdf_to_documents(pdf_path)
        if not documents:
            raise RuntimeError(
                f"PDF text extraction produced no content from: {pdf_path}"
            )
        # Save for future runs (fast JSONL path)
        _save_as_jsonl(documents, jsonl_path)
        logger.info(
            "JSONL saved. Subsequent runs will use the fast JSONL path.\n"
            "Re-run  python -m src.main --index  to rebuild indices."
        )
        return documents

    # ── PRIMARY: Load from JSONL ───────────────────────────────────────────
    documents: List[Document] = []
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)
                skipped += 1
                continue

            content = record.get("content", "").strip()
            if not content:
                logger.warning(
                    "Skipping empty-content record at line %d", line_num
                )
                skipped += 1
                continue

            inner_meta = record.get("metadata", {})
            page_s = record.get("page_start", 0)

            doc = Document(
                page_content=content,
                metadata={
                    "document_id":  record.get("document_id", "unknown"),
                    "page_start":   page_s,
                    "page_end":     record.get("page_end", 0),
                    "content_type": record.get("content_type", "structured_text"),
                    "source":       SOURCE_LABEL,
                    "section_name": _section_name(page_s),
                    "char_length":  inner_meta.get("char_length", len(content)),
                    "has_tables":   inner_meta.get("has_tables", False),
                },
            )
            documents.append(doc)

    logger.info(
        "Loaded %d documents from JSONL (%d skipped).", len(documents), skipped
    )
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = load_documents()
    for d in docs:
        print(
            f"[p{d.metadata['page_start']}–{d.metadata['page_end']}] "
            f"section={d.metadata.get('section_name','')} | "
            f"len={len(d.page_content)} chars | "
            f"tables={d.metadata['has_tables']}"
        )
