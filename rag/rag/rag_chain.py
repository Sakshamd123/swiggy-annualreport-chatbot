"""
rag_chain.py — LangChain Runnable pipeline for grounded Swiggy QA.

Pipeline (LangChain Runnables)
------------------------------
  query_input (str)
      → QueryClassifier        [RunnableLambda]  — rule-based, no LLM
      → HybridRetriever        [RunnableLambda]  — FAISS + BM25 + RRF
      → Reranker               [RunnableLambda]  — optional, modular
      → ContextValidator       [RunnableLambda]  — guardrails / fallback
      → PromptTemplate + LLM   [LCEL chain]      — single Gemini call
      → PostProcessor          [RunnableLambda]  — citations + confidence

The pipeline is built lazily: FAISS and BM25 indices are loaded once and
injected at construction time, not per query.
"""

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from rag.config import GEMINI_MODEL, GEMINI_TEMP, GOOGLE_API_KEY, FINAL_TOP_K
from rag.rag.prompt import RAG_PROMPT
from rag.rag.guardrails import validate_and_build_context, FALLBACK_ANSWER, NO_CONTEXT_SNIPPET
from rag.retrieval.hybrid_retriever import hybrid_retrieve
from rag.retrieval.reranker import rerank
from rag.utils.schema import RAGResponse, ClassifiedQuery, QueryCategory
from rag.query_rewriter import QueryRewriter
from rag.utils.parent_context import expand_to_parent_context

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Query Classifier — lightweight, rule-based, zero LLM calls
# ═══════════════════════════════════════════════════════════════════════════

_FINANCIAL_KEYWORDS = {
    "revenue", "ebitda", "net loss", "profit", "loss", "earnings",
    "cash flow", "balance sheet", "financial", "income", "expense",
    "crore", "lakh", "million", "ipo", "gross", "margin", "tax",
    "depreciation", "amortisation", "amortization", "dividend",
    "equity", "liability", "asset",
    "standalone", "consolidated", "statement", "turnover", "operating",
    "₹", "inr", "growth", "quarter", "fy", "annual",
}

_GOVERNANCE_KEYWORDS = {
    # Roles — short forms
    "board", "director", "ceo", "cfo", "coo", "secretary", "committee",
    # Roles — expanded forms (so "Chief Financial Officer" matches)
    "chief executive officer", "chief financial officer",
    "chief operating officer", "managing director", "executive director",
    "independent director", "non-executive", "chairman", "chairperson",
    # Governance actions / bodies
    "governance", "remuneration", "nomination",
    "audit committee", "shareholder", "agm", "egm",
    "compliance", "regulatory", "sebi", "roc", "mca", "irdai",
    # Auditor terms — moved from FINANCIAL
    "auditor", "auditors", "statutory auditor", "independent auditor",
    "audit report", "board of directors",
    # Share / capital structure terms (Fix 3)
    "preference share", "preference shares", "convertible",
    "ccps", "series k", "series j", "compulsorily convertible",
}


def classify_query(query: str) -> ClassifiedQuery:
    """
    Rule-based query classifier.  No LLM involved.

    Returns a ClassifiedQuery with:
      category          : "financial" | "governance" | "general"
      boost_table_chunks: True for financial queries
    """
    q_lower = query.lower()

    fin_hits = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in q_lower)
    gov_hits = sum(1 for kw in _GOVERNANCE_KEYWORDS if kw in q_lower)

    if fin_hits > gov_hits and fin_hits > 0:   # strict > so ties default to governance
        category: QueryCategory     = "financial"
        boost_table_chunks: bool    = True
    elif gov_hits > 0:
        category                    = "governance"
        boost_table_chunks          = False
    else:
        category                    = "general"
        boost_table_chunks          = False

    logger.debug(
        "Query classified: category=%s boost=%s (fin_hits=%d gov_hits=%d)",
        category, boost_table_chunks, fin_hits, gov_hits,
    )
    return ClassifiedQuery(
        query=query,
        category=category,
        boost_table_chunks=boost_table_chunks,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RAG Chain Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_rag_chain(
    faiss_store: FAISS,
    bm25_index: BM25Okapi,
    bm25_chunks: List[Document],
) -> Any:  # returns a callable that accepts str → RAGResponse
    """
    Construct the full LangChain Runnable pipeline.

    The chain is closed over the pre-loaded indices so they are not
    reloaded on every query.

    Parameters
    ----------
    faiss_store  : loaded FAISS vectorstore
    bm25_index   : loaded BM25Okapi instance
    bm25_chunks  : corpus Documents aligned with bm25_index

    Returns
    -------
    A callable (RunnableLambda) that accepts a plain string query
    and returns a RAGResponse dataclass.
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Create a .env file with GOOGLE_API_KEY=<your-key>."
        )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMP,
        google_api_key=GOOGLE_API_KEY,
    )

    # ── Internal LLM sub-chain (prompt → LLM → string parser) ────────────
    llm_chain = RAG_PROMPT | llm | StrOutputParser()

    # ── Query Rewriter (lazy-init — shared across requests) ──────────────
    _rewriter = QueryRewriter()

    # ── Step functions ────────────────────────────────────────────────────

    def step_rewrite_and_classify(query: str) -> ClassifiedQuery:
        """
        1. Rewrite the query for retrieval (Gemini Flash, no-op on failure).
        2. Classify the REWRITTEN query for category + table-boost flag.
        3. Store the ORIGINAL query so it reaches the LLM answer step.
        """
        rewritten = _rewriter.rewrite(query)
        classified = classify_query(rewritten)
        # Carry original query forward so the LLM answers in the user's words
        classified.query = rewritten          # used for retrieval
        classified.__dict__["original_query"] = query  # preserved for LLM
        return classified

    def step_retrieve(classified: ClassifiedQuery) -> Dict[str, Any]:
        retrieved = hybrid_retrieve(
            query=classified.query,          # rewritten query → better recall
            faiss_store=faiss_store,
            bm25_index=bm25_index,
            bm25_chunks=bm25_chunks,
            boost_table_chunks=classified.boost_table_chunks,
        )
        reranked = rerank(classified.query, retrieved, top_k=FINAL_TOP_K)

        # ── Parent Document Retrieval ────────────────────────────────────
        # Expand each retrieved chunk to include surrounding section context.
        # Guardrails run AFTER this expansion (as required).
        expanded = expand_to_parent_context(reranked, bm25_chunks)

        # The original user query is used for the LLM answer generation.
        original_query = classified.__dict__.get("original_query", classified.query)

        return {
            "query":     original_query,     # original → LLM answer
            "category":  classified.category,
            "retrieved": expanded,           # expanded context → guardrails
        }

    def step_validate(state: Dict[str, Any]) -> Dict[str, Any]:
        is_valid, context_text, confidence, source_pages, snippet = (
            validate_and_build_context(
                state["retrieved"],
                query    = state.get("query", ""),
                category = state.get("category", "general"),
            )
        )
        state["is_valid"]       = is_valid
        state["context_text"]   = context_text
        state["confidence"]     = confidence
        state["source_pages"]   = source_pages
        state["snippet"]        = snippet
        return state

    def step_generate(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state["is_valid"]:
            state["answer"] = FALLBACK_ANSWER
            state["fallback"] = True
            return state

        # Single LLM call — the only LLM usage in the entire pipeline
        answer = llm_chain.invoke({
            "context":  state["context_text"],
            "question": state["query"],
        })
        state["answer"]   = answer
        state["fallback"] = False
        return state

    # Phrases that indicate the LLM could not find the answer in context
    _NOT_FOUND_MARKERS = (
        "not found in the annual report",
        "cannot be found",
        "not available in the",
        "does not contain",
        "no information",
        "not mentioned",
    )

    def step_postprocess(state: Dict[str, Any]) -> RAGResponse:
        answer    = state.get("answer", "")
        is_fallback = state.get("fallback", False)

        # If LLM itself declared "Not found", override metadata so the UI
        # never shows bogus High-confidence citations for empty answers.
        llm_not_found = any(
            marker in answer.lower() for marker in _NOT_FOUND_MARKERS
        )

        if is_fallback or llm_not_found:
            final_confidence   = "Low"
            final_source_pages = "N/A"
            final_snippet      = NO_CONTEXT_SNIPPET
            final_fallback     = True
        else:
            final_confidence   = state.get("confidence", "Low")
            final_source_pages = state.get("source_pages", "N/A")
            final_snippet      = state.get("snippet", "")
            final_fallback     = False

        return RAGResponse(
            answer=answer,
            source_pages=final_source_pages,
            confidence=final_confidence,
            context_snippet=final_snippet,
            retrieved_chunks=[
                type("RC", (), {
                    "content":  doc.page_content,
                    "metadata": doc.metadata,
                    "score":    score,
                })()
                for doc, score in state.get("retrieved", [])
            ],
            query_category=state.get("category", "general"),
            fallback=final_fallback,
        )

    # ── Assemble Runnable sequence ─────────────────────────────────────────
    pipeline = (
        RunnableLambda(step_rewrite_and_classify)   # Query Rewriting → Classify
        | RunnableLambda(step_retrieve)              # Retrieval + Parent Expansion
        | RunnableLambda(step_validate)              # Guardrails (post-expansion)
        | RunnableLambda(step_generate)              # Single Gemini call
        | RunnableLambda(step_postprocess)
    )

    return pipeline
