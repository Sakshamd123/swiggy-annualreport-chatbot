import re
import html
import logging
from fastapi import APIRouter, HTTPException

from app.schemas import QueryRequest, QueryResponse

# Import RAG application logic
from rag.main import _ensure_indices
from rag.rag.rag_chain import build_rag_chain

logger = logging.getLogger(__name__)

router = APIRouter()

# Global pipeline instance
rag_pipeline = None

# ---------------------------------------------------------------------------
# Display-time snippet cleaner
# Applied ONLY to context_snippet before it is sent to the UI.
# The RAG pipeline, LLM context, and indices are completely untouched.
# ---------------------------------------------------------------------------

_BOILERPLATE = re.compile(
    r"<\s*this\s*space\s*has\s*been\s*intentionally\s*left\s*blank\s*>",
    re.IGNORECASE,
)
_IMAGE_TAG   = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_CAMEL_SPLIT = re.compile(r"(?<=[a-z₹])(?=[A-Z])|(?<=[0-9])(?=[A-Z][a-z])")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK = re.compile(r"\n{3,}")

_KNOWN_RUNONS = [
    (re.compile(r"Notesto(Standalone|Consolidated)", re.IGNORECASE),
     lambda m: f"Notes to {m.group(1)}"),
    (re.compile(r"FinancialStatements", re.IGNORECASE),
     lambda m: "Financial Statements"),
    (re.compile(r"\(formerlyknownas", re.IGNORECASE),
     lambda m: "(formerly known as "),
    (re.compile(r"AsatMarch", re.IGNORECASE),
     lambda m: "As at March"),
    (re.compile(r"AllamountinMillion", re.IGNORECASE),
     lambda m: "All amount in Million"),
]


def _clean_snippet(text: str) -> str:
    """
    Clean only the context snippet shown in the UI.
    Does NOT affect the RAG pipeline, the LLM prompt, or any indices.
    """
    if not text:
        return text

    # 1. Decode ALL HTML entities (&lt; &gt; &amp; &#39; &nbsp; …)
    text = html.unescape(text)

    # 2. Remove <!-- image --> placeholders
    text = _IMAGE_TAG.sub("", text)

    # 3. Remove blank-page boilerplate
    text = _BOILERPLATE.sub("", text)

    # 4. Fix specific known PDF run-on patterns first (order matters)
    for pattern, replacement in _KNOWN_RUNONS:
        text = pattern.sub(replacement, text)

    # 5. Split remaining CamelCase run-ons on prose lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Leave markdown table rows untouched
        if stripped.startswith("|") or stripped.startswith("|-"):
            cleaned.append(line.rstrip())
        else:
            line = _CAMEL_SPLIT.sub(" ", line)
            cleaned.append(_MULTI_SPACE.sub(" ", line).rstrip())

    text = "\n".join(cleaned)

    # 6. Collapse excessive blank lines
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------

def init_pipeline():
    global rag_pipeline
    logger.info("Initializing RAG pipeline from FastAPI...")
    try:
        faiss_store, bm25_index, bm25_chunks = _ensure_indices(force_rebuild=False)
        rag_pipeline = build_rag_chain(faiss_store, bm25_index, bm25_chunks)
        logger.info("RAG Pipeline ready.")
    except Exception as exc:
        logger.error("Failed to initialize RAG pipeline: %s", exc)
        raise


# ---------------------------------------------------------------------------
# API route
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, summary="Ask a question about the Swiggy Annual Report")
async def query_report(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")
    try:
        response = rag_pipeline.invoke(request.question)

        category_str   = response.query_category.value if hasattr(response.query_category, 'value') else str(response.query_category)
        confidence_str = response.confidence.value     if hasattr(response.confidence,     'value') else str(response.confidence)

        # Clean the snippet at display time only — pipeline/index untouched
        clean_context = _clean_snippet(response.context_snippet or "")

        return QueryResponse(
            answer=response.answer,
            confidence=confidence_str,
            category=category_str,
            source_pages=response.source_pages,
            context_snippet=clean_context,
        )
    except Exception as e:
        logger.error("Error during query: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")
