"""
evaluation/baselines.py

Four retrieval baselines for Nya(AI) evaluation:
  B1: BM25Plus only
  B2: Dense semantic only
  B3: Hybrid (existing retrieval.py logic)
  B4: Full six-signal reranker (existing reranker.py logic)

All baselines return ranked lists of historical CASE IDs (not chunk IDs).
Chunk-to-case aggregation uses best-chunk-score-per-case strategy.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Allow running standalone from evaluation/ dir
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "..", "backend")
sys.path.insert(0, _BACKEND)

import numpy as np
from rank_bm25 import BM25Plus
from sqlalchemy.orm import Session

from app import models
from app.services.embeddings import get_embedding_provider, cosine_similarity
from app.services.retrieval import retrieve_historical_chunks, aggregate_by_historical_case
from app.services.reranker import rerank_candidates

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "mock")


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were",
    "for", "by", "with", "that", "this", "it", "as", "at", "be", "are", "from",
    "has", "have", "had", "which", "who", "whom", "not", "but", "his", "her",
    "their", "its", "he", "she", "they", "will", "shall", "may", "such",
}


def _keywords(text: str):
    import re
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _get_all_historical_chunks(db: Session):
    """Return all chunks that belong to historical cases."""
    return db.query(models.Chunk).filter(
        models.Chunk.historical_case_id.isnot(None)
    ).all()


def _chunks_to_case_ranking(
    chunks: List[models.Chunk],
    scores: List[float],
    per_case_cap: int = 3,
) -> List[str]:
    """
    Aggregate chunk scores to case level.
    Returns ordered list of historical_case_ids (best first).
    Applies per-case diversification cap.
    """
    case_best: Dict[str, float] = {}
    case_count: Dict[str, int] = {}
    for chunk, score in zip(chunks, scores):
        hc_id = chunk.historical_case_id
        if hc_id is None:
            continue
        cnt = case_count.get(hc_id, 0)
        if cnt >= per_case_cap:
            continue
        case_count[hc_id] = cnt + 1
        if hc_id not in case_best or score > case_best[hc_id]:
            case_best[hc_id] = score

    ranked = sorted(case_best.items(), key=lambda x: x[1], reverse=True)
    return [hc_id for hc_id, _ in ranked]


@dataclass
class BaselineResult:
    baseline: str
    query_id: str
    ranked_cases: List[str]          # ordered historical_case_ids
    latency_ms: float
    extra: Dict[str, Any]


def run_bm25_only(
    db: Session,
    query_id: str,
    query_text: str,
    top_k: int = 15,
) -> BaselineResult:
    """B1: BM25Plus only — no dense embeddings, no reranking."""
    t0 = time.perf_counter()

    chunks = _get_all_historical_chunks(db)
    if not chunks:
        return BaselineResult("bm25_only", query_id, [], 0.0, {})

    tokenized_query = list(_keywords(query_text))
    tokenized_corpus = [list(_keywords(c.text)) for c in chunks]
    bm25 = BM25Plus(tokenized_corpus)
    raw_scores = bm25.get_scores(tokenized_query)
    clamped = [max(0.0, float(s)) for s in raw_scores]

    latency_ms = (time.perf_counter() - t0) * 1000
    ranked = _chunks_to_case_ranking(chunks, clamped, per_case_cap=3)
    return BaselineResult("bm25_only", query_id, ranked[:top_k], latency_ms, {})


def run_dense_only(
    db: Session,
    query_id: str,
    query_text: str,
    top_k: int = 15,
) -> BaselineResult:
    """B2: Dense semantic similarity only — no BM25, no reranking."""
    t0 = time.perf_counter()

    chunks = _get_all_historical_chunks(db)
    if not chunks:
        return BaselineResult("dense_only", query_id, [], 0.0, {})

    provider = get_embedding_provider()
    query_vec = provider.embed([query_text])[0]

    sem_scores = [
        cosine_similarity(query_vec, c.embedding or [])
        for c in chunks
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    ranked = _chunks_to_case_ranking(chunks, sem_scores, per_case_cap=3)
    return BaselineResult("dense_only", query_id, ranked[:top_k], latency_ms, {})


def run_hybrid(
    db: Session,
    query_id: str,
    query_text: str,
    top_k: int = 15,
    alpha: float = 0.70,
) -> BaselineResult:
    """
    B3: Hybrid (alpha * dense + (1-alpha) * BM25).
    Reuses the existing retrieve_historical_chunks logic.
    """
    t0 = time.perf_counter()

    retrieved = retrieve_historical_chunks(db, query_text, top_k=top_k * 3, per_case_cap=3)
    ranked_cases = []
    seen = set()
    for r in retrieved:
        hc_id = r.chunk.historical_case_id
        if hc_id and hc_id not in seen:
            seen.add(hc_id)
            ranked_cases.append(hc_id)

    latency_ms = (time.perf_counter() - t0) * 1000
    return BaselineResult("hybrid", query_id, ranked_cases[:top_k], latency_ms,
                          {"alpha": alpha})


def run_full_reranker(
    db: Session,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
    weights: Optional[Dict[str, float]] = None,
) -> BaselineResult:
    """
    B4: Full six-signal reranker.
    If query_case is None, creates a temporary Case object for reranking.
    """
    t0 = time.perf_counter()

    if query_case is None:
        query_case = models.Case(
            title=query_text[:80],
            facts=query_text,
            issues=query_text,
        )

    candidates = retrieve_historical_chunks(db, query_text, top_k=top_k * 4, per_case_cap=3)
    reranked = rerank_candidates(db, query_case, candidates, top_k=top_k, weights=weights)

    seen = set()
    ranked_cases = []
    for r in reranked:
        hc_id = r.historical_case.id
        if hc_id not in seen:
            seen.add(hc_id)
            ranked_cases.append(hc_id)

    latency_ms = (time.perf_counter() - t0) * 1000
    return BaselineResult("full_reranker", query_id, ranked_cases[:top_k], latency_ms,
                          {"weights": weights or "default"})


def run_all_baselines(
    db: Session,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> Dict[str, BaselineResult]:
    """Run all four baselines on one query and return results dict."""
    return {
        "bm25_only":    run_bm25_only(db, query_id, query_text, top_k),
        "dense_only":   run_dense_only(db, query_id, query_text, top_k),
        "hybrid":       run_hybrid(db, query_id, query_text, top_k),
        "full_reranker": run_full_reranker(db, query_id, query_text, query_case, top_k),
    }
