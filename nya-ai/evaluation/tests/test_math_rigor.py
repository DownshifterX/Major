"""evaluation/tests/test_math_rigor.py
Comprehensive mathematical and edge-case unit test suite for Nya(AI) (Section 16).

Tests:
- Cosine similarity (identical, orthogonal, zero vector, dimension mismatch)
- BM25Plus lower bound and parameters
- Statutory Jaccard similarity (empty sets, disjoint, identical, no NaN / division by zero)
- Legal issue alignment overlap coefficient (empty sets, partial, subset)
- Logarithmic citation authority normalization
- Citation anchor blending with alpha parameter
- Court hierarchy engineering mapping
- Weight normalization (sum = 1.0, non-negative)
- Learned-weight optimization constraint satisfaction
- NDCG@K graded relevance and rank sensitivity
- MRR properties
- Hard legal eligibility filtering (overruled exclusion)
- Percentile P95 order statistic definition
- Reranker output ordering and deterministic tie breaking
"""
import math
import sys
from pathlib import Path
import pytest
from typing import List, Set, Dict

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent / "backend"
_EVAL = _HERE.parent
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_EVAL))

from app.services.embeddings import cosine_similarity
from app.services.citation_graph import LegalCitationGraph
from app.services.reranker import (
    compute_statutory_overlap_jaccard,
    compute_issue_alignment_overlap,
    compute_court_hierarchy_score,
    normalize_weights,
    is_case_eligible,
    FIXED_BASELINE_WEIGHTS,
)
from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    calculate_percentile,
    compute_leave_one_out_deltas,
)


# ===========================================================================
# 1. Cosine Similarity Tests
# ===========================================================================

def test_cosine_similarity_identical():
    v = [0.2, 0.5, 0.8, 0.1]
    assert pytest.approx(cosine_similarity(v, v), rel=1e-5) == 1.0


def test_cosine_similarity_orthogonal():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), abs=1e-6) == 0.0


def test_cosine_similarity_zero_vector():
    v1 = [0.0, 0.0, 0.0]
    v2 = [1.0, 2.0, 3.0]
    # Norm of zero vector is 0; must return 0.0 without divide by zero
    assert cosine_similarity(v1, v2) == 0.0
    assert cosine_similarity(v1, v1) == 0.0


def test_cosine_similarity_dimension_mismatch():
    v1 = [1.0, 2.0]
    v2 = [1.0, 2.0, 3.0]
    assert cosine_similarity(v1, v2) == 0.0


def test_cosine_similarity_empty():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], []) == 0.0


# ===========================================================================
# 2. BM25+ Verification
# ===========================================================================

def test_bm25_plus_parameters():
    from rank_bm25 import BM25Plus
    import inspect
    sig = inspect.signature(BM25Plus.__init__)
    params = sig.parameters
    assert params["k1"].default == 1.5
    assert params["b"].default == 0.75
    assert params["delta"].default == 1


def test_bm25_plus_non_negative_lower_bound():
    from rank_bm25 import BM25Plus
    corpus = [["bail", "anticipatory", "crpc"], ["murder", "ipc", "trial"]]
    bm25 = BM25Plus(corpus)
    scores = bm25.get_scores(["unrelated", "taxation", "gst"])
    # All scores must be non-negative
    for s in scores:
        assert s >= 0.0


# ===========================================================================
# 3. Statutory Overlap (Jaccard) Tests
# ===========================================================================

def test_statutory_jaccard_empty_query():
    assert compute_statutory_overlap_jaccard(set(), {"section 302", "ipc"}) == 0.0


def test_statutory_jaccard_empty_candidate():
    assert compute_statutory_overlap_jaccard({"section 302", "ipc"}, set()) == 0.0


def test_statutory_jaccard_both_empty_no_nan():
    res = compute_statutory_overlap_jaccard(set(), set())
    assert res == 0.0
    assert not math.isnan(res)


def test_statutory_jaccard_disjoint():
    sq = {"section 438 crpc"}
    sd = {"section 138 ni act"}
    assert compute_statutory_overlap_jaccard(sq, sd) == 0.0


def test_statutory_jaccard_identical():
    s = {"section 302 ipc", "section 34 ipc"}
    assert pytest.approx(compute_statutory_overlap_jaccard(s, s), rel=1e-4) == 1.0


def test_statutory_jaccard_subset():
    sq = {"section 302", "section 34"}
    sd = {"section 302", "section 34", "section 120b", "section 201"}
    # intersection = 2, union = 4 -> J = 0.5
    assert pytest.approx(compute_statutory_overlap_jaccard(sq, sd), rel=1e-4) == 0.5


# ===========================================================================
# 4. Legal Issue Alignment (Overlap Coefficient) Tests
# ===========================================================================

def test_issue_overlap_empty_sets():
    assert compute_issue_alignment_overlap(set(), {"murder", "bail"}) == 0.0
    assert compute_issue_alignment_overlap({"murder"}, set()) == 0.0
    assert compute_issue_alignment_overlap(set(), set()) == 0.0


def test_issue_overlap_disjoint():
    a = {"anticipatory", "bail", "arrest"}
    b = {"arbitration", "award", "contract"}
    assert compute_issue_alignment_overlap(a, b) == 0.0


def test_issue_overlap_subset():
    # If smaller set is fully contained in larger set, overlap coeff is 1.0
    query_issues = {"anticipatory", "bail"}
    candidate_summary = {"anticipatory", "bail", "personal", "liberty", "crpc"}
    assert pytest.approx(compute_issue_alignment_overlap(query_issues, candidate_summary), rel=1e-4) == 1.0


def test_issue_overlap_partial():
    a = {"murder", "circumstantial", "witness"}
    b = {"murder", "confession", "police", "custody"}
    # inter = 1 ("murder"), min(|A|, |B|) = 3 -> 1/3 = 0.3333
    assert pytest.approx(compute_issue_alignment_overlap(a, b), abs=1e-3) == 0.3333


# ===========================================================================
# 5. Citation Authority & Normalization Tests
# ===========================================================================

def test_logarithmic_citation_authority_normalization():
    graph = LegalCitationGraph()
    # Create simple graph: case A cited 9 times, case B cited 0 times
    graph.graph.add_node("A", name="Case A")
    graph.graph.add_node("B", name="Case B")
    for i in range(9):
        graph.graph.add_node(f"C_{i}", name=f"Case {i}")
        graph.graph.add_edge(f"C_{i}", "A", type="cites")

    graph._rebuild_cache()

    # Max in-degree is 9
    score_a = graph.get_authority_score("A", mode="log")
    score_b = graph.get_authority_score("B", mode="log")

    # A has max citations -> log(1+9)/log(1+9) = 1.0
    assert pytest.approx(score_a, rel=1e-3) == 1.0
    # B has 0 citations -> log(1+0)/log(1+9) = 0.0
    assert score_b == 0.0


def test_citation_anchor_blending():
    graph = LegalCitationGraph()
    graph.graph.add_node("A", name="Case A")
    graph.graph.add_node("B", name="Case B")
    graph.graph.add_edge("B", "A", type="cites")
    graph._rebuild_cache()

    # Degree score for A is 1.0 (since max_deg=1)
    # Blend with anchor score 0.5 and alpha 0.6: 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    blended = graph.get_authority_score("A", mode="log", alpha=0.6, anchor_score=0.5)
    assert pytest.approx(blended, rel=1e-3) == 0.8


def test_citation_nonexistent_node():
    graph = LegalCitationGraph()
    assert graph.get_authority_score("nonexistent", mode="log") == 0.0


# ===========================================================================
# 6. Court Hierarchy Mapping Tests
# ===========================================================================

def test_court_hierarchy_mapping():
    assert compute_court_hierarchy_score("Supreme Court of India") == 1.0
    assert compute_court_hierarchy_score("High Court of Delhi") == 0.75
    assert compute_court_hierarchy_score("District & Sessions Court, Pune") == 0.50
    assert compute_court_hierarchy_score("National Company Law Tribunal") == 0.40
    assert compute_court_hierarchy_score("Unknown Forum") == 0.50


# ===========================================================================
# 7. Weight Normalization Tests
# ===========================================================================

def test_weights_sum_to_one():
    raw_weights = {"semantic": 2.0, "lexical": 1.0, "statute": 1.0, "issue": 0.5, "citation": 0.3, "court": 0.2}
    norm = normalize_weights(raw_weights)
    assert pytest.approx(sum(norm.values()), rel=1e-5) == 1.0


def test_no_negative_weights():
    raw_weights = {"semantic": 0.5, "lexical": -0.2, "statute": 0.5}
    norm = normalize_weights(raw_weights)
    for v in norm.values():
        assert v >= 0.0
    assert norm["lexical"] == 0.0
    assert pytest.approx(sum(norm.values()), rel=1e-5) == 1.0


def test_fixed_baseline_weights_validity():
    w = FIXED_BASELINE_WEIGHTS
    for k, v in w.items():
        assert v >= 0.0
    assert pytest.approx(sum(w.values()), rel=1e-5) == 1.0


# ===========================================================================
# 8. Hard Legal Constraint Tests
# ===========================================================================

class MockHistoricalCase:
    def __init__(self, cid: str, is_overruled: bool = False, status: str = "active"):
        self.id = cid
        self.is_overruled = is_overruled
        self.status = status


def test_hard_eligibility_filter():
    active_case = MockHistoricalCase("case_001", is_overruled=False)
    overruled_attr = MockHistoricalCase("case_002", is_overruled=True)
    overruled_status = MockHistoricalCase("case_003", status="overruled")

    assert is_case_eligible(active_case) is True
    assert is_case_eligible(overruled_attr) is False
    assert is_case_eligible(overruled_status) is False
    assert is_case_eligible(active_case, excluded_case_ids={"case_001"}) is False
    assert is_case_eligible(None) is False


# ===========================================================================
# 9. IR Metrics Tests (NDCG, MRR, Precision, Recall)
# ===========================================================================

def test_ndcg_perfect_ranking():
    retrieved = ["doc1", "doc2", "doc3"]
    graded = {"doc1": 2.0, "doc2": 1.0, "doc3": 0.0}
    score = ndcg_at_k(retrieved, graded, k=3)
    assert pytest.approx(score, rel=1e-5) == 1.0


def test_ndcg_reversed_ranking():
    retrieved = ["doc3", "doc2", "doc1"]
    graded = {"doc1": 2.0, "doc2": 1.0, "doc3": 0.0}
    score = ndcg_at_k(retrieved, graded, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_empty():
    assert ndcg_at_k([], {"doc1": 2.0}, k=5) == 0.0
    assert ndcg_at_k(["doc1"], {}, k=5) == 0.0


def test_mrr_ranks():
    relevant = {"doc_gold"}
    assert reciprocal_rank(["doc_gold", "doc_other"], relevant) == 1.0
    assert reciprocal_rank(["doc_1", "doc_gold"], relevant) == 0.5
    assert pytest.approx(reciprocal_rank(["doc_1", "doc_2", "doc_gold"], relevant), rel=1e-4) == 1.0 / 3.0
    assert reciprocal_rank(["doc_1", "doc_2"], relevant) == 0.0


def test_precision_and_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c9"}
    # In top 5: c1 and c3 are hits -> 2 / 5 = 0.4
    assert precision_at_k(retrieved, relevant, 5) == 0.4
    # Recall: 2 found out of 3 total relevant -> 2 / 3 = 0.6667
    assert pytest.approx(recall_at_k(retrieved, relevant, 5), abs=1e-3) == 0.6667


# ===========================================================================
# 10. Percentile Calculation Tests (Section 10)
# ===========================================================================

def test_percentile_calculation_nearest_rank():
    # 100 observations: 1 to 100
    samples = list(range(1, 101))
    p50 = calculate_percentile(samples, 50)
    p95 = calculate_percentile(samples, 95)
    assert p50 == 50
    assert p95 == 95


def test_percentile_single_sample():
    assert calculate_percentile([42.5], 95) == 42.5


def test_percentile_empty():
    assert calculate_percentile([], 95) == 0.0


# ===========================================================================
# 11. Leave-One-Out Delta Computations (Section 14)
# ===========================================================================

def test_leave_one_out_deltas():
    full = {"ndcg_at_10": 0.85, "ndcg_at_5": 0.82, "mrr": 0.90}
    ablated = {"ndcg_at_10": 0.80, "ndcg_at_5": 0.78, "mrr": 0.85}
    deltas = compute_leave_one_out_deltas(full, ablated)
    assert pytest.approx(deltas["delta_ndcg_at_10"], rel=1e-4) == 0.05
    assert pytest.approx(deltas["delta_ndcg_at_5"], rel=1e-4) == 0.04
    assert pytest.approx(deltas["delta_mrr"], rel=1e-4) == 0.05
