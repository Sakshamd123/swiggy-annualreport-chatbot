"""
query_rewriter.py — LLM-based query rewriting for improved retrieval recall.

Design
------
• Uses Gemini Flash (same model as the main RAG chain) to rewrite vague or
  short user queries into retrieval-optimised search strings.
• The rewritten query is used ONLY for FAISS + BM25 retrieval.
• The ORIGINAL user query is always used for the final LLM answer generation.
• If rewriting fails for any reason, falls back to the original query silently
  — this ensures the enhancement is strictly additive and never breaks the
  existing pipeline.

Integration point
-----------------
Called inside step_classify() in rag_chain.py BEFORE the classifier and
retrieval steps. Classifier and retrieval use the rewritten query; the
original query propagates forward for LLM answer generation.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import GOOGLE_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = """\
You are a retrieval query optimizer for a financial document search engine.

Your ONLY job is to rewrite a user question into a short, keyword-rich search
query that will retrieve the most relevant passages from an annual report.

Rules:
- Output ONLY the rewritten search query. No explanation, no preamble.
- Keep it concise (10–20 words maximum).
- Add relevant financial or governance keywords if the question is vague.
- Do NOT answer the question.
- Do NOT change the meaning or intent.
- If the query is already specific and keyword-rich, return it unchanged.
- Use terminology found in annual reports (e.g. "revenue from operations",
  "standalone", "consolidated", "board of directors", "FY24", "March 2024").
"""

_REWRITE_HUMAN = "User question: {question}"

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REWRITE_SYSTEM),
    ("human",  _REWRITE_HUMAN),
])

# Threshold: if rewritten text is longer than this it is probably an
# unintentional answer, so fall back to original query.
_MAX_REWRITE_LENGTH = 200


# ---------------------------------------------------------------------------
# QueryRewriter class
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Wraps a Gemini Flash call to produce a retrieval-optimised query string.

    Usage
    -----
        rewriter = QueryRewriter()
        retrieval_query = rewriter.rewrite("How is Swiggy doing?")
        # → "Swiggy revenue net loss financial performance FY24 annual report"
    """

    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set — QueryRewriter cannot initialise."
            )
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,                  # deterministic
            google_api_key=GOOGLE_API_KEY,
        )
        self._chain = _REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        Rewrite *query* for better retrieval recall.

        Returns the rewritten string, or *query* unchanged if rewriting
        fails or produces a suspiciously long response.

        Parameters
        ----------
        query : original user question

        Returns
        -------
        str   : rewritten query (used for retrieval only)
        """
        query = query.strip()
        if not query:
            return query

        try:
            rewritten = self._chain.invoke({"question": query}).strip()

            # Safety: if the model returned something very long it probably
            # tried to answer the question — fall back to original.
            if not rewritten or len(rewritten) > _MAX_REWRITE_LENGTH:
                logger.warning(
                    "QueryRewriter returned suspicious output (len=%d). "
                    "Falling back to original query.",
                    len(rewritten),
                )
                return query

            logger.info(
                "QueryRewriter | original=%r  →  rewritten=%r",
                query, rewritten,
            )
            return rewritten

        except Exception as exc:
            logger.warning(
                "QueryRewriter failed (%s). Falling back to original query.",
                exc,
            )
            return query
