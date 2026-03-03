"""
benchmark_v2.py — Comprehensive Accuracy & Performance Benchmark for Swiggy RAG Backend

New Features vs v1:
 - Expanded query set (15 queries spanning all 3 categories)
 - Per-category breakdown (financial / governance / general)
 - Per-query verbose debug output: retrieved pages, chunk types, scores
 - Top-3 / Top-5 / Top-10 accuracy (where N > retrieved set allows)
 - MRR per category
 - Guardrail hit rate (how often MIN_RELEVANCE_SCORE is triggered)
 - Page-range match diagnostic: exact vs. within-range
 - UTF-8 output safe for Windows terminal
"""

import sys
import os
import time
import io

# UTF-8 safe output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path fix for direct execution ────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag.config import FAISS_INDEX_DIR, BM25_INDEX_PATH, MIN_RELEVANCE_SCORE, FINAL_TOP_K
from rag.indexing.vector_store import get_or_build_faiss
from rag.indexing.bm25_index import get_or_build_bm25
from rag.retrieval.hybrid_retriever import hybrid_retrieve
from rag.rag.rag_chain import classify_query
from rag.ingestion.jsonl_loader import load_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.chunker import chunk_documents
from rag.rag.guardrails import validate_and_build_context


# ── EXTENDED GROUND TRUTH ────────────────────────────────────────────────────
# 15 queries spanning financial, governance, and general categories
# expected_pages: list of pages that CONTAIN the answer
BENCHMARK_QUERIES = [

    # ── FINANCIAL ─────────────────────────────────────────────────────────────
    {
        "id": "F01",
        "query": "What was Swiggy's standalone revenue from operations for the year ended March 31, 2024?",
        "expected_pages": [3, 4, 46, 47],  # Board's report summary p3-4, P&L p46-47
        "category": "financial",
        "notes": "Standalone P&L — Revenue from operations: ₹63,722.98 million"
    },
    {
        "id": "F02",
        "query": "What was Swiggy's net loss after tax on a standalone basis for FY2024?",
        "expected_pages": [3, 4, 46, 47, 60, 61],
        "category": "financial",
        "notes": "Standalone net loss: ₹18,880.32 million"
    },
    {
        "id": "F03",
        "query": "What is the total consolidated revenue from operations for FY24?",
        "expected_pages": [3, 4, 109, 110, 151, 152],
        "category": "financial",
        "notes": "Consolidated revenue from operations: ₹112,474 million"
    },
    {
        "id": "F04",
        "query": "What are the total assets of Swiggy as of March 31, 2024 on a standalone basis?",
        "expected_pages": [43, 44, 45, 46],
        "category": "financial",
        "notes": "Standalone Balance Sheet — total assets: ₹108,046.16 million"
    },
    {
        "id": "F05",
        "query": "What is the cash and cash equivalents at the end of FY2024 in the standalone financial statements?",
        "expected_pages": [46, 55, 56, 57, 60],
        "category": "financial",
        "notes": "Cash and cash equivalents: ₹7,871.26 million"
    },
    {
        "id": "F06",
        "query": "How much did Swiggy spend on advertising and sales promotion in FY2024?",
        "expected_pages": [46, 47, 48],
        "category": "financial",
        "notes": "Advertising and sales promotion expense: ₹20,380.09 million"
    },
    {
        "id": "F07",
        "query": "What are the delivery and related charges expense for year ended March 31, 2024?",
        "expected_pages": [46, 47, 48],
        "category": "financial",
        "notes": "Delivery and related charges: ₹33,510.90 million"
    },

    # ── GOVERNANCE ────────────────────────────────────────────────────────────
    {
        "id": "G01",
        "query": "Who are the statutory auditors of Swiggy Limited?",
        "expected_pages": [1, 2, 16, 17, 28, 29, 30, 31],
        "category": "governance",
        "notes": "B S R & Co. LLP, Chartered Accountants"
    },
    {
        "id": "G02",
        "query": "Who is the CFO (Chief Financial Officer) of Swiggy?",
        "expected_pages": [1, 2, 16, 17, 28, 29],
        "category": "governance",
        "notes": "Rahul Bothra - CFO"
    },
    {
        "id": "G03",
        "query": "How many board meetings were held during FY24?",
        "expected_pages": [16, 17, 18, 19, 20],
        "category": "governance",
        "notes": "6 board meetings held during the year"
    },
    {
        "id": "G04",
        "query": "How many Series K Compulsorily Convertible Preference Shares were issued by Swiggy?",
        "expected_pages": [1, 2, 15, 16, 17],
        "category": "governance",
        "notes": "1,08,000 Series K CCPS of face value INR 10,000 each"
    },
    {
        "id": "G05",
        "query": "Who is the Chairman of the Board of Directors of Swiggy?",
        "expected_pages": [1, 2, 16, 17],
        "category": "governance",
        "notes": "Anand Kripalu - Independent Director Chairman"
    },

    # ── GENERAL ───────────────────────────────────────────────────────────────
    {
        "id": "N01",
        "query": "What is Swiggy's core business model and services?",
        "expected_pages": [1, 2, 3, 4, 46, 60],
        "category": "general",
        "notes": "Food delivery, quick commerce (Instamart), Dineout, SteppinOut"
    },
    {
        "id": "N02",
        "query": "When was Swiggy converted from a private to a public limited company?",
        "expected_pages": [3, 4, 5, 6, 7, 8],
        "category": "general",
        "notes": "April 10, 2024 — fresh certificate of incorporation from RoC"
    },
    {
        "id": "N03",
        "query": "What subsidiaries does Swiggy have as of March 31, 2024?",
        "expected_pages": [3, 4, 5, 6, 7, 8, 16],
        "category": "general",
        "notes": "Scootsy Logistics, Supr Infotech Solutions, Lynks Logistics (step-down)"
    },
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def page_hit(doc, expected_set: set) -> bool:
    """True if any expected page falls inside the chunk's [page_start, page_end]."""
    p_start = doc.metadata.get("page_start", -1)
    p_end   = doc.metadata.get("page_end",   -1)
    return any(p_start <= ep <= p_end for ep in expected_set)


def load_indices():
    print("[1/2] Loading Documents & Indices...")
    docs   = clean_documents(load_documents())
    chunks = chunk_documents(docs)
    faiss_store              = get_or_build_faiss(chunks)
    bm25_index, bm25_chunks  = get_or_build_bm25(chunks)
    print(f"      Loaded {len(chunks)} chunks | FAISS index | BM25 index\n")
    return faiss_store, bm25_index, bm25_chunks


# ── MAIN BENCHMARK ────────────────────────────────────────────────────────────

def run_benchmark(faiss_store, bm25_index, bm25_chunks):
    print("[2/2] Running Extended Retrieval Benchmark v2")
    print("=" * 75)

    total         = len(BENCHMARK_QUERIES)
    top1_hits     = 0
    top3_hits     = 0
    top5_hits     = 0
    mrr_sum       = 0.0
    guardrail_hits= 0
    missed_queries= []
    total_latency = 0.0

    # per-category counters
    cat_stats = {
        "financial":  {"total": 0, "top1": 0, "top5": 0, "mrr": 0.0},
        "governance": {"total": 0, "top1": 0, "top5": 0, "mrr": 0.0},
        "general":    {"total": 0, "top1": 0, "top5": 0, "mrr": 0.0},
    }

    for item in BENCHMARK_QUERIES:
        qid      = item["id"]
        query    = item["query"]
        expected = set(item["expected_pages"])
        category = item["category"]
        notes    = item.get("notes", "")

        t0 = time.time()

        # 1. Classify
        cq = classify_query(query)

        # 2. Retrieve
        results = hybrid_retrieve(
            query              = cq.query,
            faiss_store        = faiss_store,
            bm25_index         = bm25_index,
            bm25_chunks        = bm25_chunks,
            boost_table_chunks = cq.boost_table_chunks,
        )

        # 3. Check guardrail
        is_valid, _, confidence, _, _ = validate_and_build_context(results)
        if not is_valid:
            guardrail_hits += 1

        elapsed_ms = (time.time() - t0) * 1000
        total_latency += elapsed_ms

        # 4. Score
        hit_rank   = None
        for rank, (doc, score) in enumerate(results, start=1):
            if page_hit(doc, expected):
                hit_rank = rank
                break

        # Update global counters
        if hit_rank == 1:
            top1_hits += 1
        if hit_rank is not None and hit_rank <= 3:
            top3_hits += 1
        if hit_rank is not None and hit_rank <= 5:
            top5_hits += 1
        if hit_rank is not None:
            mrr_sum += 1.0 / hit_rank
        else:
            missed_queries.append(qid)

        # Update per-category counters
        cat = cat_stats[category]
        cat["total"] += 1
        if hit_rank == 1:
            cat["top1"] += 1
        if hit_rank is not None and hit_rank <= 5:
            cat["top5"] += 1
        if hit_rank is not None:
            cat["mrr"] += 1.0 / hit_rank

        # 5. Per-query output
        rank_str = str(hit_rank) if hit_rank else "MISSED"
        print(f"\n[{qid}] [{category.upper():<10}] {query[:65]}...")
        print(f"   Classified as  : {cq.category} (boost_table={cq.boost_table_chunks})")
        print(f"   Guardrail      : {'PASS' if is_valid else 'BLOCKED (' + str(confidence) + ')'}")
        print(f"   Hit Rank       : {rank_str}  |  Latency: {elapsed_ms:.0f} ms")
        print(f"   Expected pages : {sorted(expected)}")

        # Show top-3 retrieved pages with scores
        print(f"   Retrieved chunks:")
        for rank, (doc, score) in enumerate(results[:5], start=1):
            ps   = doc.metadata.get("page_start", "?")
            pe   = doc.metadata.get("page_end",   "?")
            ct   = doc.metadata.get("chunk_type",  "?")
            hit  = "[HIT]" if page_hit(doc, expected) else "     "
            print(f"      Rank {rank}: {hit} Pages {ps}-{pe} | {ct:<12} | score={score:.4f}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    top1_acc  = (top1_hits / total) * 100
    top3_acc  = (top3_hits / total) * 100
    top5_acc  = (top5_hits / total) * 100
    mrr       = mrr_sum / total
    avg_lat   = total_latency / total

    print("\n" + "=" * 75)
    print("  BENCHMARK RESULTS SUMMARY  (benchmark_v2.py)")
    print("=" * 75)
    print(f"  Total Queries         : {total}")
    print(f"  Top-1 Accuracy        : {top1_acc:.1f}%   ({top1_hits}/{total})")
    print(f"  Top-3 Accuracy        : {top3_acc:.1f}%   ({top3_hits}/{total})")
    print(f"  Top-5 Accuracy        : {top5_acc:.1f}%   ({top5_hits}/{total})")
    print(f"  Mean Reciprocal Rank  : {mrr:.4f}")
    print(f"  Avg Latency per Query : {avg_lat:.0f} ms")
    print(f"  Guardrail Blocks      : {guardrail_hits}/{total} queries blocked by min_score={MIN_RELEVANCE_SCORE}")
    print(f"  Missed Queries        : {missed_queries if missed_queries else 'None'}")

    print("\n  --- Per-Category Breakdown ---")
    print(f"  {'Category':<12} {'Queries':>7} {'Top-1':>7} {'Top-5':>7} {'MRR':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for cat_name, cs in cat_stats.items():
        if cs["total"] == 0:
            continue
        c_top1 = cs["top1"] / cs["total"] * 100
        c_top5 = cs["top5"] / cs["total"] * 100
        c_mrr  = cs["mrr"]  / cs["total"]
        print(f"  {cat_name:<12} {cs['total']:>7} {c_top1:>6.0f}% {c_top5:>6.0f}% {c_mrr:>8.4f}")

    print("=" * 75)


if __name__ == "__main__":
    faiss_store, bm25_index, bm25_chunks = load_indices()
    run_benchmark(faiss_store, bm25_index, bm25_chunks)
