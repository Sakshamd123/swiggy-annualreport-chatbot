import logging
from fastapi import APIRouter, HTTPException
from contextlib import asynccontextmanager

from app.schemas import QueryRequest, QueryResponse

# Import RAG application logic
from rag.main import _ensure_indices
from rag.rag.rag_chain import build_rag_chain

logger = logging.getLogger(__name__)

router = APIRouter()

# Global pipeline instance
rag_pipeline = None

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

@router.post("/query", response_model=QueryResponse, summary="Ask a question about the Swiggy Annual Report")
async def query_report(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")
    try:
        response = rag_pipeline.invoke(request.question)
        
        # Determine string representation of enums if needed
        category_str = response.query_category.value if hasattr(response.query_category, 'value') else str(response.query_category)
        confidence_str = response.confidence.value if hasattr(response.confidence, 'value') else str(response.confidence)

        return QueryResponse(
            answer=response.answer,
            confidence=confidence_str,
            category=category_str,
            source_pages=response.source_pages,
            context_snippet=response.context_snippet
        )
    except Exception as e:
        logger.error("Error during query: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")
