"""evaluation/ablation.py
Ablation study module for Nya(AI) multi-signal precedent reranker.

Supports:
1. Leave-One-Out Ablation (Section 14):
   Quantifies the individual marginal contribution of each signal:
       Delta NDCG@10(signal) = NDCG@10(all signals) - NDCG@10(without signal)
       Delta MRR(signal)     = MRR(all signals) - MRR(without signal)
   Configurations:
   - Full (all six signals)
   - minus semantic
   - minus lexical
   - minus statutory
   - minus issue
   - minus citation
   - minus court

2. Incremental Ablation (A0 to A5):
   Progressively adds signals from dense-only up to the full 6-signal baseline.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from sqlalchemy.orm import Session
from app import models
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import rerank_candidates, normalize_weights, FIXED_BASELINE_WEIGHTS


# ---------------------------------------------------------------------------
# Leave-One-Out Ablation Configurations (Section 14)
# ---------------------------------------------------------------------------

def _build_leave_one_out_configs() -> Dict[str, Dict[str, float]]:
    base = dict(FIXED_BASELINE_WEIGHTS)
    configs: Dict[str, Dict[str, float]] = {
        "all_six_signals": normalize_weights(base),
    }
    for signal in ["semantic", "lexical", "statute", "issue", "citation", "court"]:
        ablated = dict(base)
        ablated[signal] = 0.0
        configs[f"minus_{signal}"] = normalize_weights(ablated)
    return configs


LEAVE_ONE_OUT_CONFIGS = _build_leave_one_out_configs()

LEAVE_ONE_OUT_DESCRIPTIONS = {
    "all_six_signals": "All six signals (Fixed MCDA baseline)",
    "minus_semantic":  "All signals minus Semantic similarity",
    "minus_lexical":   "All signals minus Lexical similarity (BM25+)",
    "minus_statute":   "All signals minus Statutory overlap",
    "minus_issue":     "All signals minus Legal issue alignment",
    "minus_citation":  "All signals minus Citation authority",
    "minus_court":     "All signals minus Court hierarchy",
}


# ---------------------------------------------------------------------------
# Incremental Ablation Configurations (A0 to A5)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "A0_dense_only": normalize_weights({
        "semantic": 1.0, "lexical": 0.0, "statute": 0.0, "issue": 0.0, "citation": 0.0, "court": 0.0,
    }),
    "A1_dense_bm25": normalize_weights({
        "semantic": 0.70, "lexical": 0.30, "statute": 0.0, "issue": 0.0, "citation": 0.0, "court": 0.0,
    }),
    "A2_dense_bm25_statute": normalize_weights({
        "semantic": 0.50, "lexical": 0.25, "statute": 0.25, "issue": 0.0, "citation": 0.0, "court": 0.0,
    }),
    "A3_dense_bm25_statute_issue": normalize_weights({
        "semantic": 0.45, "lexical": 0.22, "statute": 0.22, "issue": 0.11, "citation": 0.0, "court": 0.0,
    }),
    "A4_dense_bm25_statute_issue_citation": normalize_weights({
        "semantic": 0.40, "lexical": 0.20, "statute": 0.20, "issue": 0.10, "citation": 0.10, "court": 0.0,
    }),
    "A5_full_system": dict(FIXED_BASELINE_WEIGHTS),
}

ABLATION_DESCRIPTIONS = {
    "A0_dense_only":                        "Dense semantic similarity only",
    "A1_dense_bm25":                        "Dense + BM25 lexical",
    "A2_dense_bm25_statute":                "Dense + BM25 + Statutory overlap",
    "A3_dense_bm25_statute_issue":          "Dense + BM25 + Statutory + Issue alignment",
    "A4_dense_bm25_statute_issue_citation": "Dense + BM25 + Statutory + Issue + Citation authority",
    "A5_full_system":                       "Full system (all 6 signals)",
}


@dataclass
class AblationResult:
    config_name: str
    description: str
    weights: Dict[str, float]
    query_id: str
    ranked_cases: List[str]
    latency_ms: float


def run_ablation_config(
    db: Session,
    config_name: str,
    query_id: str,
    query_text: str,
    weights_dict: Dict[str, float],
    description: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> AblationResult:
    """Run one ablation configuration for one query."""
    t0 = time.perf_counter()

    if query_case is None:
        query_case = models.Case(
            title=query_text[:80],
            facts=query_text,
            issues=query_text,
        )

    candidates = retrieve_historical_chunks(db, query_text, top_k=top_k * 4, per_case_cap=3)
    reranked = rerank_candidates(db, query_case, candidates, top_k=top_k, weights=weights_dict)

    seen = set()
    ranked_cases = []
    for r in reranked:
        hc_id = r.historical_case.id
        if hc_id not in seen:
            seen.add(hc_id)
            ranked_cases.append(hc_id)

    latency_ms = (time.perf_counter() - t0) * 1000

    return AblationResult(
        config_name=config_name,
        description=description,
        weights=weights_dict,
        query_id=query_id,
        ranked_cases=ranked_cases[:top_k],
        latency_ms=latency_ms,
    )


def run_all_ablations(
    db: Session,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> Dict[str, AblationResult]:
    """Run incremental ablation configs (A0-A5) for one query."""
    results = {}
    for config_name, w in ABLATION_CONFIGS.items():
        results[config_name] = run_ablation_config(
            db=db,
            config_name=config_name,
            query_id=query_id,
            query_text=query_text,
            weights_dict=w,
            description=ABLATION_DESCRIPTIONS.get(config_name, config_name),
            query_case=query_case,
            top_k=top_k,
        )
    return results


def run_leave_one_out_ablations(
    db: Session,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> Dict[str, AblationResult]:
    """Run Leave-One-Out ablation configs for one query."""
    results = {}
    for config_name, w in LEAVE_ONE_OUT_CONFIGS.items():
        results[config_name] = run_ablation_config(
            db=db,
            config_name=config_name,
            query_id=query_id,
            query_text=query_text,
            weights_dict=w,
            description=LEAVE_ONE_OUT_DESCRIPTIONS.get(config_name, config_name),
            query_case=query_case,
            top_k=top_k,
        )
    return results
