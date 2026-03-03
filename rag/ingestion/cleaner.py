"""
cleaner.py — Markdown-aware text normalisation for Swiggy financial docs.

Rules
-----
* Strip HTML image placeholders: <!-- image -->
* Collapse runs of blank lines (>2) to a single blank line
* Normalise whitespace inside non-table lines
* Preserve markdown table rows (lines containing |) completely intact
* Normalise currency symbols to a canonical form (₹ / INR kept as-is)
* Remove Unicode control characters that may confuse tokenisers
"""

import re
import logging
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)

# ── simple patterns ──────────────────────────────────────────────────────────
_IMAGE_TAG    = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_CTRL_CHARS   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK  = re.compile(r"\n{3,}")
_WHITESPACE   = re.compile(r"[ \t]+")
_AMP_ENTITY   = re.compile(r"&amp;")


def _is_table_row(line: str) -> bool:
    """Return True if this line is part of a markdown table."""
    stripped = line.strip()
    return stripped.startswith("|") or stripped.startswith("|-")


def clean_text(text: str) -> str:
    """
    Normalise a single markdown string without destroying table structure.
    """
    # 1. Remove HTML image placeholders
    text = _IMAGE_TAG.sub("", text)

    # 2. Fix HTML entities that Docling sometimes leaves in
    text = _AMP_ENTITY.sub("&", text)

    # 3. Strip invisible control characters (keep \n \t)
    text = _CTRL_CHARS.sub("", text)

    # 4. Process line by line — preserve table rows untouched
    lines = text.split("\n")
    cleaned: List[str] = []
    for line in lines:
        if _is_table_row(line):
            # preserve table row exactly (just strip trailing space)
            cleaned.append(line.rstrip())
        else:
            # collapse internal whitespace for prose lines
            cleaned.append(_WHITESPACE.sub(" ", line).rstrip())

    text = "\n".join(cleaned)

    # 5. Collapse excessive blank lines
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Apply clean_text to every document's page_content in place.
    Returns the same list (mutated) for memory efficiency.
    """
    for doc in documents:
        original_len = len(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
        new_len = len(doc.page_content)
        if original_len - new_len > 500:
            logger.debug(
                "Cleaned doc p%s–%s: %d → %d chars",
                doc.metadata.get("page_start"),
                doc.metadata.get("page_end"),
                original_len,
                new_len,
            )
    logger.info("Cleaned %d documents.", len(documents))
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = """<!-- image -->

    ## Revenue from operations

    |  Year | Amount |
    |-------|--------|
    | 2024  | ₹12,000 Mn |

    Some   extra   spaces   here.
    """
    print(clean_text(sample))
