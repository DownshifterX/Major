"""evaluation/performance.py
Performance and Latency Benchmarking for each stage of the Nya(AI) pipeline (Section 10 & 11).

Measures:
  - semantic_retrieval      (Dense query embedding + cosine similarity over corpus)
  - bm25_retrieval          (Tokenization + BM25Plus scoring over corpus)
  - candidate_generation    (Stage 1 hybrid pooling, scoring, deduplication, diversification)
  - feature_extraction      (Extracting the 6 normalized features for candidate pool)
  - citation_scoring        (Logarithmic in-degree authority lookup from precomputed cache)
  - reranking               (Additive utility scoring and candidate sorting)
  - complete_query_latency  (End-to-end: query text -> hybrid pool -> 6 features -> reranked cases)

Percentile Convention (Section 10):
  Order-statistic nearest-rank: on sorted observations x_(1) <= x_(2) <= ... <= x_(N):
      k = max(0, int(len(sorted_s) * 0.95) - 1)
      P95 = x_(k)

  P95 is NOT the "worst-case" latency (which is max_ms).
  It represents the latency threshold below which approximately 95% of observations
  fall under this order-statistic definition (Dean & Barroso, 2013; Jain, 1991).
"""
import os
import sys
import time
import json
import csv
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
_RESULTS_DIR = _HERE / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_BACKEND))

from sqlalchemy.orm import Session
from rank_bm25 import BM25Plus

from app import models
from app.services.embeddings import get_embedding_provider, cosine_similarity
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import (
    rerank_candidates,
    extract_candidate_features,
    _tokenize_text,
    get_reranker_weights,
)
from app.services.citation_graph import get_citation_graph

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was",
    "were", "for", "by", "with", "that", "this", "it", "as", "at",
}


def _keywords(text: str):
    import re
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


@dataclass
class LatencyStats:
    stage: str
    samples: List[float]
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def __post_init__(self):
        if self.samples:
            sorted_s = sorted(self.samples)
            self.mean_ms = round(statistics.mean(sorted_s), 2)
            self.median_ms = round(statistics.median(sorted_s), 2)
            idx_95 = max(0, int(len(sorted_s) * 0.95) - 1)
            self.p95_ms = round(sorted_s[idx_95], 2)
            self.min_ms = round(sorted_s[0], 2)
            self.max_ms = round(sorted_s[-1], 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "n_samples": len(self.samples),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "percentile_convention": "nearest_rank_order_statistic_k=floor(0.95*N)-1",
        }


def benchmark_semantic_retrieval(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark dense query embedding + vector scan over chunk embeddings."""
    provider = get_embedding_provider()
    chunks = db.query(models.Chunk).filter(models.Chunk.historical_case_id.isnot(None)).all()
    latencies = []

    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter()
            query_vec = provider.embed([q])[0]
            _ = [cosine_similarity(query_vec, c.embedding or []) for c in chunks]
            latencies.append((time.perf_counter() - t0) * 1000)

    return LatencyStats("semantic_retrieval", latencies)


def benchmark_bm25_retrieval(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark tokenization + BM25Plus score computation over indexed corpus."""
    chunks = db.query(models.Chunk).filter(models.Chunk.historical_case_id.isnot(None)).all()
    tokenized_corpus = [list(_keywords(c.text)) for c in chunks]
    latencies = []

    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter()
            bm25 = BM25Plus(tokenized_corpus)
            _ = bm25.get_scores(list(_keywords(q)))
            latencies.append((time.perf_counter() - t0) * 1000)

    return LatencyStats("bm25_retrieval", latencies)


def benchmark_candidate_generation(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark Stage 1 hybrid pooling, scoring, deduplication, and diversification."""
    latencies = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter()
            _ = retrieve_historical_chunks(db, q, top_k=20)
            latencies.append((time.perf_counter() - t0) * 1000)
    return LatencyStats("candidate_generation", latencies)


def benchmark_feature_extraction(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark extracting the 6 normalized legal signals for candidates."""
    graph = get_citation_graph(db)
    latencies = []

    for _ in range(n_runs):
        for q in queries:
            candidates = retrieve_historical_chunks(db, q, top_k=20)
            q_provisions = set(prov.lower() for prov in ["section 302", "section 438", "article 21"])
            q_issues = _tokenize_text(q)
            t0 = time.perf_counter()
            for cand in candidates:
                hc = db.query(models.HistoricalCase).get(cand.chunk.historical_case_id)
                if hc:
                    _ = extract_candidate_features(
                        cand=cand,
                        hc=hc,
                        target_provisions=q_provisions,
                        target_issues_tokens=q_issues,
                        graph=graph,
                    )
            latencies.append((time.perf_counter() - t0) * 1000)

    return LatencyStats("feature_extraction", latencies)


def benchmark_citation_scoring(db: Session, n_runs: int = 5) -> LatencyStats:
    """Benchmark precomputed logarithmic citation authority lookup."""
    graph = get_citation_graph(db)
    cases = db.query(models.HistoricalCase).limit(30).all()
    latencies = []

    for _ in range(n_runs):
        for hc in cases:
            t0 = time.perf_counter()
            _ = graph.get_authority_score(hc.id, mode="log")
            latencies.append((time.perf_counter() - t0) * 1000)

    return LatencyStats("citation_scoring", latencies)


def benchmark_reranking(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark Stage 2 multi-signal reranking given candidate pool."""
    latencies = []
    for _ in range(n_runs):
        for q in queries:
            candidates = retrieve_historical_chunks(db, q, top_k=20)
            qcase = models.Case(title=q[:60], facts=q, issues=q)
            t0 = time.perf_counter()
            _ = rerank_candidates(db, qcase, candidates, top_k=10, mode="fixed")
            latencies.append((time.perf_counter() - t0) * 1000)
    return LatencyStats("reranking", latencies)


def benchmark_complete_query_latency(db: Session, queries: List[str], n_runs: int = 2) -> LatencyStats:
    """Benchmark end-to-end pipeline: raw query -> hybrid candidate pool -> 6 features -> ranked cases."""
    latencies = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter()
            candidates = retrieve_historical_chunks(db, q, top_k=25)
            qcase = models.Case(title=q[:60], facts=q, issues=q)
            _ = rerank_candidates(db, qcase, candidates, top_k=10, mode="fixed")
            latencies.append((time.perf_counter() - t0) * 1000)
    return LatencyStats("complete_query_latency", latencies)


def run_full_benchmark(
    db: Session,
    queries: List[str],
    quick: bool = False,
) -> List[LatencyStats]:
    """Runs latency benchmarks across all stages of the Nya(AI) architecture."""
    n_runs = 1 if quick else 2
    test_queries = queries[:2] if quick else queries[:4]
    results: List[LatencyStats] = []

    print("  [perf] Benchmarking semantic retrieval...")
    results.append(benchmark_semantic_retrieval(db, test_queries, n_runs=n_runs))

    print("  [perf] Benchmarking BM25 retrieval...")
    results.append(benchmark_bm25_retrieval(db, test_queries, n_runs=n_runs))

    print("  [perf] Benchmarking candidate generation (pooling)...")
    results.append(benchmark_candidate_generation(db, test_queries, n_runs=n_runs))

    print("  [perf] Benchmarking feature extraction...")
    results.append(benchmark_feature_extraction(db, test_queries, n_runs=n_runs))

    print("  [perf] Benchmarking citation scoring...")
    results.append(benchmark_citation_scoring(db, n_runs=max(2, n_runs)))

    print("  [perf] Benchmarking reranking...")
    results.append(benchmark_reranking(db, test_queries, n_runs=n_runs))

    print("  [perf] Benchmarking complete query latency...")
    results.append(benchmark_complete_query_latency(db, test_queries, n_runs=n_runs))

    return results


def run_and_save_performance(quick: bool = False):
    """Executes the benchmark and saves performance.json and performance.csv."""
    from app.database import SessionLocal

    queries = [
        "Anticipatory bail under Section 438 CrPC economic offences",
        "Murder Section 302 IPC eyewitness reliability circumstantial evidence",
        "Fundamental rights Article 21 personal liberty due process",
    ]

    db = SessionLocal()
    try:
        stats = run_full_benchmark(db, queries, quick=quick)
        rows = [s.to_dict() for s in stats]

        for out_path in [_RESULTS_DIR / "performance.json", _HERE.parent / "performance.json"]:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print(f"  Saved performance metrics to: {out_path.name}")

        csv_path = _RESULTS_DIR / "performance.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "n_samples", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "percentile_convention"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved performance CSV to: {csv_path.name}")

        return rows
    finally:
        db.close()


if __name__ == "__main__":
    quick_mode = "--quick" in sys.argv
    run_and_save_performance(quick=quick_mode)
