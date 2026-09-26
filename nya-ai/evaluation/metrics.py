"""evaluation/metrics.py
Standard Information Retrieval (IR) evaluation metrics for Nya(AI) retrieval evaluation.

Academic Foundations:
- Järvelin & Kekäläinen (2002): NDCG@K (Cumulated Gain-based Evaluation of IR Techniques, ACM TOIS)
- Voorhees (1999): Mean Reciprocal Rank (MRR, TREC-8)
- Manning, Raghavan, & Schütze (2008): Precision@K and Recall@K (Introduction to Information Retrieval)
- Jain (1991) & Dean & Barroso (2013): Latency percentiles (P50, P95)

Implements:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K (with graded relevance)
- Hit Rate@K
- Percentile calculation (P50, P95)
- Leave-one-out ablation delta computations
"""
import math
from typing import List, Set, Dict, Optional, Any


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if not retrieved or not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / min(k, len(top_k))


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant items found in top-k."""
    if not retrieved or not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant item; 0 if none found."""
    if not retrieved or not relevant:
        return 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], graded_relevance: Dict[str, float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K (Järvelin & Kekäläinen, 2002).

    graded_relevance: dict mapping case_id -> relevance score
        2.0 = highly relevant (direct statutory & legal match)
        1.0 = partially relevant (broad domain / related precedent)
        0.0 = not relevant (or absent)

    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
    """
    if not retrieved or not graded_relevance or k <= 0:
        return 0.0

    def dcg(ranked: List[str]) -> float:
        total = 0.0
        for i, item in enumerate(ranked[:k], start=1):
            rel = graded_relevance.get(item, 0.0)
            if rel > 0.0:
                total += (2.0 ** rel - 1.0) / math.log2(i + 1)
        return total

    dcg_val = dcg(retrieved)

    # Ideal DCG: sort all known relevant items by graded score descending
    ideal_order = sorted(graded_relevance.keys(), key=lambda x: graded_relevance[x], reverse=True)
    idcg_val = dcg(ideal_order)

    if idcg_val <= 0.0:
        return 0.0
    return min(1.0, dcg_val / idcg_val)


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """1 if at least one relevant item is in top-k, else 0."""
    if not retrieved or not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


def compute_all_metrics(
    retrieved: List[str],
    relevant: Set[str],
    highly_relevant: Optional[Set[str]] = None,
    partially_relevant: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Compute the full suite of IR metrics for one query.

    Parameters
    ----------
    retrieved       : Ordered list of retrieved case IDs (best first)
    relevant        : All relevant case IDs (union of highly + partially)
    highly_relevant : Subset of highly relevant cases (rel=2)
    partially_relevant: Subset of partially relevant cases (rel=1)

    Returns a dict of all metric names -> float values.
    """
    highly_relevant = highly_relevant or set()
    partially_relevant = partially_relevant or (relevant - highly_relevant)

    # Build graded relevance dict
    graded: Dict[str, float] = {}
    for case_id in highly_relevant:
        graded[case_id] = 2.0
    for case_id in partially_relevant:
        if case_id not in graded:
            graded[case_id] = 1.0

    return {
        "precision_at_1":  round(precision_at_k(retrieved, relevant, 1), 4),
        "precision_at_3":  round(precision_at_k(retrieved, relevant, 3), 4),
        "precision_at_5":  round(precision_at_k(retrieved, relevant, 5), 4),
        "precision_at_10": round(precision_at_k(retrieved, relevant, 10), 4),
        "recall_at_5":     round(recall_at_k(retrieved, relevant, 5), 4),
        "recall_at_10":    round(recall_at_k(retrieved, relevant, 10), 4),
        "mrr":             round(reciprocal_rank(retrieved, relevant), 4),
        "ndcg_at_5":       round(ndcg_at_k(retrieved, graded, 5), 4),
        "ndcg_at_10":      round(ndcg_at_k(retrieved, graded, 10), 4),
        "hit_rate_at_1":   round(hit_rate_at_k(retrieved, relevant, 1), 4),
        "hit_rate_at_5":   round(hit_rate_at_k(retrieved, relevant, 5), 4),
        "hit_rate_at_10":  round(hit_rate_at_k(retrieved, relevant, 10), 4),
    }


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average per-query metrics across all queries.
    Returns macro-averaged results.
    """
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    averaged = {}
    for k in keys:
        vals = [m[k] for m in per_query_metrics if k in m]
        averaged[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return averaged


def calculate_percentile(samples: List[float], p: float) -> float:
    """
    Calculates the p-th percentile (0 <= p <= 100) using nearest-rank order statistic.

    Definition:
        On sorted observations x_(1) <= x_(2) <= ... <= x_(N):
        k = max(0, int(len(sorted_s) * (p / 100.0)) - 1)
        Percentile_p = x_(k)

    Clarification:
        P95 is NOT the "worst-case" latency (which is max).
        It represents the latency threshold below which approximately 95% of
        observations fall under this order-statistic definition.
    """
    if not samples:
        return 0.0
    sorted_s = sorted(samples)
    n = len(sorted_s)
    if n == 1:
        return round(sorted_s[0], 2)
    idx = max(0, min(n - 1, int(n * (p / 100.0)) - 1))
    return round(sorted_s[idx], 2)


def compute_leave_one_out_deltas(
    full_metrics: Dict[str, float],
    ablated_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Computes Leave-One-Out signal contribution deltas (Section 14):
        Delta NDCG@10 = NDCG@10(all signals) - NDCG@10(without signal)
        Delta MRR     = MRR(all signals) - MRR(without signal)
        Delta NDCG@5  = NDCG@5(all signals) - NDCG@5(without signal)

    A positive delta indicates that the removed signal contributed positively
    to ranking effectiveness.
    """
    return {
        "delta_ndcg_at_10": round(full_metrics.get("ndcg_at_10", 0.0) - ablated_metrics.get("ndcg_at_10", 0.0), 4),
        "delta_ndcg_at_5":  round(full_metrics.get("ndcg_at_5", 0.0) - ablated_metrics.get("ndcg_at_5", 0.0), 4),
        "delta_mrr":         round(full_metrics.get("mrr", 0.0) - ablated_metrics.get("mrr", 0.0), 4),
    }
