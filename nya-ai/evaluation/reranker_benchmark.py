"""evaluation/reranker_benchmark.py
Comparative Benchmark Framework for Nya(AI) Retrieval & Reranking Architecture (Section 3).

Compares:
  A. BM25 only
  B. Semantic retrieval only
  C. BM25 + semantic retrieval (Hybrid)
  D. Fixed six-signal MCDA baseline
  E. Learned six-signal reranker (NDCG@K optimized)

Metrics:
  - MRR
  - NDCG@5
  - NDCG@10
  - Recall@5
  - Recall@10
  - Mean Latency (ms)

Produces:
  - evaluation/results/reranker_comparison.json
  - evaluation/reranker_comparison.json
  - Human-readable comparison table
"""
import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
_RESULTS_DIR = _HERE / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_HERE))

from app.database import SessionLocal
from app import models
from baselines import run_bm25_only, run_dense_only, run_hybrid, run_full_reranker
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import rerank_candidates, get_reranker_weights, FIXED_BASELINE_WEIGHTS
from metrics import compute_all_metrics, aggregate_metrics

SYSTEM_NAMES = [
    "A_bm25_only",
    "B_semantic_only",
    "C_hybrid_bm25_semantic",
    "D_fixed_mcda_baseline",
    "E_learned_reranker",
]

SYSTEM_DESCRIPTIONS = {
    "A_bm25_only":              "BM25Plus lexical retrieval only",
    "B_semantic_only":          "Dense embedding semantic similarity only",
    "C_hybrid_bm25_semantic":   "Hybrid retrieval (0.70 Dense + 0.30 BM25+)",
    "D_fixed_mcda_baseline":    "Fixed six-signal MCDA baseline [0.35, 0.20, 0.20, 0.10, 0.08, 0.07]",
    "E_learned_reranker":       "Learned six-signal reranker (NDCG@K validation-optimized)",
}


def run_single_query_comparison(
    db: Any,
    query_dict: Dict[str, Any],
    ground_truth: Dict[str, Dict[str, Any]],
    top_k: int = 15,
) -> Dict[str, Dict[str, Any]]:
    """Runs all 5 systems for one query and returns metrics per system."""
    qid = query_dict["query_id"]
    q_text = query_dict["query_text"]
    gt = ground_truth.get(qid, {})
    relevant = set(gt.get("relevant_cases", []))
    highly_relevant = set(gt.get("highly_relevant_cases", []))
    partially_relevant = set(gt.get("partially_relevant_cases", []))

    # Build query case object
    qcase = models.Case(
        title=q_text[:80],
        facts=q_text,
        issues=" ".join(query_dict.get("legal_issues", [])) or q_text,
        provisions=query_dict.get("statutory_provisions", []),
    )

    query_results: Dict[str, Dict[str, Any]] = {}

    # System A: BM25 only
    res_a = run_bm25_only(db, qid, q_text, top_k=top_k)
    m_a = compute_all_metrics(res_a.ranked_cases, relevant, highly_relevant, partially_relevant)
    m_a["latency_ms"] = res_a.latency_ms
    query_results["A_bm25_only"] = m_a

    # System B: Semantic only
    res_b = run_dense_only(db, qid, q_text, top_k=top_k)
    m_b = compute_all_metrics(res_b.ranked_cases, relevant, highly_relevant, partially_relevant)
    m_b["latency_ms"] = res_b.latency_ms
    query_results["B_semantic_only"] = m_b

    # System C: Hybrid (BM25 + Semantic)
    res_c = run_hybrid(db, qid, q_text, top_k=top_k)
    m_c = compute_all_metrics(res_c.ranked_cases, relevant, highly_relevant, partially_relevant)
    m_c["latency_ms"] = res_c.latency_ms
    query_results["C_hybrid_bm25_semantic"] = m_c

    # Shared candidate pool for rerankers
    candidates = retrieve_historical_chunks(db, q_text, top_k=top_k * 4, per_case_cap=3)

    # System D: Fixed Six-Signal MCDA Baseline
    t0_d = time.perf_counter()
    reranked_d = rerank_candidates(db, qcase, candidates, top_k=top_k, mode="fixed")
    lat_d = (time.perf_counter() - t0_d) * 1000 + res_c.latency_ms
    ranked_d = [r.historical_case.id for r in reranked_d]
    m_d = compute_all_metrics(ranked_d, relevant, highly_relevant, partially_relevant)
    m_d["latency_ms"] = lat_d
    query_results["D_fixed_mcda_baseline"] = m_d

    # System E: Learned Six-Signal Reranker
    t0_e = time.perf_counter()
    reranked_e = rerank_candidates(db, qcase, candidates, top_k=top_k, mode="learned")
    lat_e = (time.perf_counter() - t0_e) * 1000 + res_c.latency_ms
    ranked_e = [r.historical_case.id for r in reranked_e]
    m_e = compute_all_metrics(ranked_e, relevant, highly_relevant, partially_relevant)
    m_e["latency_ms"] = lat_e
    query_results["E_learned_reranker"] = m_e

    return query_results


def run_full_reranker_comparison(
    queries: Optional[List[Dict[str, Any]]] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Runs the 5-system comparative benchmark across all queries."""
    with open(_HERE / "queries.json", "r", encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]

    with open(_HERE / "ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)["ground_truth"]
        ground_truth = {item["query_id"]: item for item in gt_data}

    eval_queries = (queries or all_queries)[:5] if quick else (queries or all_queries)

    print(f"\n=======================================================")
    print(f" NYA(AI) 5-SYSTEM COMPARATIVE RETRIEVAL BENCHMARK")
    print(f"=======================================================")
    print(f" Queries evaluated: {len(eval_queries)}")

    per_system_query_metrics: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in SYSTEM_NAMES
    }

    db = SessionLocal()
    try:
        for i, q in enumerate(eval_queries, 1):
            qid = q["query_id"]
            print(f"  [{i}/{len(eval_queries)}] Evaluating {qid}...")
            q_res = run_single_query_comparison(db, q, ground_truth)
            for sys_name in SYSTEM_NAMES:
                per_system_query_metrics[sys_name].append(q_res[sys_name])
    finally:
        db.close()

    # Aggregate results per system
    comparison_summary: Dict[str, Any] = {}
    table_rows = []

    print(f"\n{'-'*95}")
    print(f"{'System':<28} | {'MRR':<8} | {'P@5':<8} | {'R@5':<8} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'Latency(ms)':<10}")
    print(f"{'-'*95}")

    for sys_name in SYSTEM_NAMES:
        agg = aggregate_metrics(per_system_query_metrics[sys_name])
        latencies = [m.get("latency_ms", 0.0) for m in per_system_query_metrics[sys_name]]
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        comparison_summary[sys_name] = {
            "description": SYSTEM_DESCRIPTIONS[sys_name],
            "mrr": agg.get("mrr", 0.0),
            "precision_at_5": agg.get("precision_at_5", 0.0),
            "recall_at_5": agg.get("recall_at_5", 0.0),
            "recall_at_10": agg.get("recall_at_10", 0.0),
            "ndcg_at_5": agg.get("ndcg_at_5", 0.0),
            "ndcg_at_10": agg.get("ndcg_at_10", 0.0),
            "avg_latency_ms": avg_lat,
        }

        print(
            f"{sys_name:<28} | "
            f"{agg.get('mrr', 0.0):<8.4f} | "
            f"{agg.get('precision_at_5', 0.0):<8.4f} | "
            f"{agg.get('recall_at_5', 0.0):<8.4f} | "
            f"{agg.get('ndcg_at_5', 0.0):<8.4f} | "
            f"{agg.get('ndcg_at_10', 0.0):<8.4f} | "
            f"{avg_lat:<10.2f}"
        )

        table_rows.append({
            "system": sys_name,
            "description": SYSTEM_DESCRIPTIONS[sys_name],
            "mrr": agg.get("mrr", 0.0),
            "precision_at_5": agg.get("precision_at_5", 0.0),
            "recall_at_5": agg.get("recall_at_5", 0.0),
            "recall_at_10": agg.get("recall_at_10", 0.0),
            "ndcg_at_5": agg.get("ndcg_at_5", 0.0),
            "ndcg_at_10": agg.get("ndcg_at_10", 0.0),
            "avg_latency_ms": avg_lat,
        })

    print(f"{'-'*95}\n")

    # Save to evaluation/results/reranker_comparison.json and evaluation/reranker_comparison.json
    output_payload = {
        "_metadata": {
            "evaluation_title": "Nya(AI) 5-System Retrieval & Reranker Comparison",
            "evaluation_mode": "QUICK" if quick else "FULL",
            "num_queries": len(eval_queries),
            "label_note": "Evaluated against corpus proxy relevance labels with honest proxy disclosures.",
        },
        "systems": comparison_summary,
    }

    for target in [_RESULTS_DIR / "reranker_comparison.json", _HERE / "reranker_comparison.json"]:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"  Saved comparison to: {target.name}")

    # Also save CSV
    csv_path = _RESULTS_DIR / "reranker_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "description", "mrr", "precision_at_5", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10", "avg_latency_ms"],
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  Saved comparison CSV to: {csv_path.name}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-system comparative benchmark.")
    parser.add_argument("--quick", action="store_true", help="Quick mode on 5 queries")
    args = parser.parse_args()

    run_full_reranker_comparison(quick=args.quick)
