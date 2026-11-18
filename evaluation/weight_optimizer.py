"""evaluation/weight_optimizer.py
Data-Driven Weight Optimization for Nya(AI) 6-Signal Precedent Reranker.

Formal Model:
    S(q, d) = sum_{j=1}^M w_j * s_hat_j(q, d)
    w* = argmax_w NDCG@K(validation_set)
    subject to:
        w_j >= 0  for all j in {1..M}
        sum_{j=1}^M w_j = 1.0

Features (M=6):
    1. Semantic similarity
    2. Lexical similarity
    3. Statutory overlap
    4. Legal issue alignment
    5. Citation authority
    6. Court hierarchy

Usage:
    python evaluation/weight_optimizer.py [--k 5] [--quick]
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
_CONFIG_DIR = _HERE.parent / "config"
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_HERE))

from app.database import SessionLocal
from app import models
from app.services.retrieval import retrieve_historical_chunks
from app.services.citation_graph import get_citation_graph
from app.services.reranker import (
    extract_candidate_features,
    is_case_eligible,
    normalize_weights,
    _tokenize_text,
    FIXED_BASELINE_WEIGHTS,
)
from metrics import ndcg_at_k

FEATURE_NAMES = ["semantic", "lexical", "statute", "issue", "citation", "court"]


class PrecomputedValidationQuery:
    """Pre-extracted candidate features and graded relevance for one query."""

    def __init__(
        self,
        query_id: str,
        candidate_ids: List[str],
        feature_matrix: np.ndarray,  # shape (N_candidates, 6)
        graded_relevance: Dict[str, float],
    ):
        self.query_id = query_id
        self.candidate_ids = candidate_ids
        self.feature_matrix = feature_matrix
        self.graded_relevance = graded_relevance

    def evaluate_ndcg(self, weights: np.ndarray, k: int = 5) -> float:
        """Vectorized computation of NDCG@K for a candidate weight vector."""
        if len(self.candidate_ids) == 0:
            return 0.0

        scores = self.feature_matrix @ weights
        # Order candidates descending by score; break ties deterministically
        ranked_indices = np.argsort(-scores, kind="mergesort")
        ranked_cases = [self.candidate_ids[i] for i in ranked_indices]
        return ndcg_at_k(ranked_cases, self.graded_relevance, k=k)


def prepare_validation_data(
    db: Any,
    queries: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
    pool_size: int = 25,
) -> List[PrecomputedValidationQuery]:
    """Precomputes candidate pools and 6-signal feature matrices for all validation queries."""
    graph = get_citation_graph(db)
    validation_queries: List[PrecomputedValidationQuery] = []

    print(f"  [optimizer] Precomputing features for {len(queries)} validation queries...")

    for q in queries:
        qid = q["query_id"]
        q_text = q["query_text"]
        gt = ground_truth.get(qid, {})

        # Build graded relevance dictionary
        graded: Dict[str, float] = {}
        for cid in gt.get("highly_relevant_cases", []):
            graded[cid] = 2.0
        for cid in gt.get("partially_relevant_cases", []):
            if cid not in graded:
                graded[cid] = 1.0

        if not graded:
            continue

        # Extract target case representations
        target_provisions = set(p.strip().lower() for p in q.get("statutory_provisions", []))
        target_issues_tokens = set()
        for issue in q.get("legal_issues", []):
            target_issues_tokens |= _tokenize_text(issue)
        target_issues_tokens |= _tokenize_text(q_text)

        # Retrieve candidates from Stage 1
        candidates = retrieve_historical_chunks(db, q_text, top_k=pool_size)

        candidate_ids = []
        feature_rows = []
        seen_cases = set()

        for cand in candidates:
            hc_id = cand.chunk.historical_case_id
            if hc_id in seen_cases:
                continue
            hc = db.query(models.HistoricalCase).get(hc_id)
            if not is_case_eligible(hc):
                continue

            feats = extract_candidate_features(
                cand=cand,
                hc=hc,
                target_provisions=target_provisions,
                target_issues_tokens=target_issues_tokens,
                graph=graph,
            )
            row = [feats[name] for name in FEATURE_NAMES]
            feature_rows.append(row)
            candidate_ids.append(hc_id)
            seen_cases.add(hc_id)

        if candidate_ids:
            val_q = PrecomputedValidationQuery(
                query_id=qid,
                candidate_ids=candidate_ids,
                feature_matrix=np.array(feature_rows, dtype=float),
                graded_relevance=graded,
            )
            validation_queries.append(val_q)

    print(f"  [optimizer] Prepared {len(validation_queries)} valid query-candidate evaluation pools.")
    return validation_queries


def optimize_weights(
    validation_queries: List[PrecomputedValidationQuery],
    k: int = 5,
    n_random_starts: int = 30,
) -> Tuple[Dict[str, float], float]:
    """
    Solves:
        w* = argmax_w (1/|Q_val|) sum_{q in Q_val} NDCG@K(q, w)
        subject to: w_j >= 0, sum w_j = 1.0

    Uses multi-start constrained optimization with simplex projection.
    """
    if not validation_queries:
        print("  [optimizer] WARNING: No validation queries. Returning baseline weights.")
        return dict(FIXED_BASELINE_WEIGHTS), 0.0

    def objective(w_vec: np.ndarray) -> float:
        # Project onto simplex: non-negative and sums to 1
        w_clean = np.maximum(0.0, w_vec)
        s = np.sum(w_clean)
        if s <= 0:
            w_norm = np.ones(6) / 6.0
        else:
            w_norm = w_clean / s

        mean_ndcg = float(np.mean([vq.evaluate_ndcg(w_norm, k=k) for vq in validation_queries]))
        # Return negative for minimization
        return -mean_ndcg

    # Initial candidate: fixed baseline
    w_init = np.array([FIXED_BASELINE_WEIGHTS[name] for name in FEATURE_NAMES], dtype=float)
    best_score = -objective(w_init)
    best_w = w_init.copy()

    # Multi-start random Dirichlet sampling over the simplex to explore multi-modal landscape
    # Seed 1950 commemorates Article 141 & adoption year of the Constitution of India
    np.random.seed(1950)
    candidate_starts = [w_init]
    for _ in range(n_random_starts):
        # Dirichlet with alpha=1.0 generates uniform samples over simplex
        candidate_starts.append(np.random.dirichlet(np.ones(6)))

    for start_w in candidate_starts:
        res = minimize(
            objective,
            start_w,
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4},
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            w_clean = np.maximum(0.0, res.x)
            best_w = w_clean / np.sum(w_clean)

    # Clean and round to 4 decimals, preserving sum = 1.0
    w_dict = {name: float(best_w[i]) for i, name in enumerate(FEATURE_NAMES)}
    normalized = normalize_weights(w_dict)
    return normalized, round(best_score, 4)


def save_learned_weights_to_config(
    learned_weights: Dict[str, float],
    best_ndcg: float,
    k: int,
    config_path: Optional[Path] = None,
):
    """Saves the optimized weights to config/reranker_weights.json."""
    cfg_file = config_path or (_CONFIG_DIR / "reranker_weights.json")
    cfg_file.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {}
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    data["learned_weights"] = {
        "_label": f"Validation-Optimized Weights (NDCG@{k} maximization)",
        "_validation_ndcg": best_ndcg,
        "_status": "Learned via constrained optimization on validation relevance judgments",
        "weights": learned_weights,
    }

    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [optimizer] Saved learned weights to: {cfg_file.resolve()}")


def run_weight_learning_pipeline(
    k: int = 5,
    quick: bool = False,
) -> Dict[str, float]:
    """End-to-end data-driven weight optimization pipeline."""
    with open(_HERE / "queries.json", "r", encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]

    with open(_HERE / "ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)["ground_truth"]
        ground_truth = {item["query_id"]: item for item in gt_data}

    # Select validation split (first 5 queries in quick mode, 10 queries in full mode)
    val_queries = all_queries[:5] if quick else all_queries[:10]

    db = SessionLocal()
    try:
        val_data = prepare_validation_data(db, val_queries, ground_truth)
        learned_w, best_ndcg = optimize_weights(val_data, k=k)
        print(f"\n[optimizer] Optimization complete!")
        print(f"  Best Validation NDCG@{k}: {best_ndcg:.4f}")
        print("  Learned Weights:")
        for name, val in learned_w.items():
            print(f"    {name:10s}: {val:.4f}")

        save_learned_weights_to_config(learned_w, best_ndcg, k=k)
        return learned_w
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Nya(AI) reranker weights on validation data.")
    parser.add_argument("--k", type=int, default=5, help="NDCG cut-off rank K (default 5)")
    parser.add_argument("--quick", action="store_true", help="Quick mode on subset of queries")
    args = parser.parse_args()

    run_weight_learning_pipeline(k=args.k, quick=args.quick)
