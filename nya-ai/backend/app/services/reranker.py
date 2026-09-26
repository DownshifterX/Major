"""Multi-Signal Legal Reranking Engine.

Formulation (Linear Multi-Attribute Additive Model):
    S(q, d) = sum_{j=1}^M w_j * s_hat_j(q, d)
    subject to:
        w_j >= 0
        sum_{j=1}^M w_j = 1.0
        s_hat_j(q, d) in [0.0, 1.0]

The six literature-motivated signals:
1. Semantic similarity      (Dense vector cosine similarity, clamped to [0, 1])
2. Lexical similarity       (BM25+ normalized token score)
3. Statutory overlap        (Jaccard similarity J(S_q, S_d) over extracted provisions)
4. Legal issue alignment    (Overlap coefficient Overlap(A, B) on legal issue tokens)
5. Citation authority       (Logarithmic in-degree dampening s_cit = log(1+deg)/log(1+max_deg))
6. Court hierarchy          (Engineering encoding of Article 141 Constitution of India)

Weight Modes:
- "fixed"   : Fixed-weight MCDA baseline [0.35, 0.20, 0.20, 0.10, 0.08, 0.07]
- "learned" : Empirical weights optimized on validation relevance judgments (NDCG@K)
"""
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from .. import models
from .retrieval import RetrievedChunk
from .citation_graph import get_citation_graph


# ---------------------------------------------------------------------------
# Baseline Weights & Court Hierarchy Constants
# ---------------------------------------------------------------------------

# Labeled as: "Fixed-weight MCDA baseline" (initial engineering heuristic, NOT literature-proven optimal)
FIXED_BASELINE_WEIGHTS: Dict[str, float] = {
    "semantic": 0.35,
    "lexical": 0.20,
    "statute": 0.20,
    "issue": 0.10,
    "citation": 0.08,
    "court": 0.07,
}

# Engineering encoding of Article 141 of Constitution of India (Supreme Court binding precedent)
DEFAULT_COURT_HIERARCHY: Dict[str, float] = {
    "supreme court": 1.0,
    "high court": 0.75,
    "district court": 0.50,
    "tribunal": 0.40,
    "default": 0.50,
}

STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were",
    "for", "by", "with", "that", "this", "it", "as", "at", "be", "are", "from",
    "has", "have", "had", "which", "who", "whom", "not", "but", "his", "her",
    "their", "its", "he", "she", "they", "will", "shall", "may", "such",
    "under", "case", "court", "vs", "versus", "appeal", "appellant", "respondent"
}


def _tokenize_text(text: str) -> Set[str]:
    """Extract lowercased alphanumeric tokens of length >= 3 excluding stopwords."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure weights are non-negative and sum to 1.0:
        w_j >= 0,  sum w_j = 1.0
    """
    cleaned = {k: max(0.0, float(v)) for k, v in w.items()}
    total = sum(cleaned.values())
    if total <= 0:
        # Fallback to uniform distribution over keys
        n = max(1, len(cleaned))
        return {k: 1.0 / n for k in cleaned}
    return {k: round(v / total, 6) for k, v in cleaned.items()}


def get_reranker_weights(
    mode: Optional[str] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Resolves active reranker weights.
    Order of precedence:
      1. Explicitly passed custom_weights
      2. Mode parameter ('fixed' vs 'learned')
      3. Environment variable RERANKER_WEIGHT_MODE ('fixed' vs 'learned')
      4. Configuration file config/reranker_weights.json
      5. Fallback to FIXED_BASELINE_WEIGHTS
    """
    if custom_weights:
        return normalize_weights(custom_weights)

    active_mode = (
        mode
        or os.getenv("RERANKER_WEIGHT_MODE")
        or "fixed"
    ).lower()

    if active_mode == "fixed":
        return dict(FIXED_BASELINE_WEIGHTS)

    # Attempt to load learned weights from config file
    config_paths = [
        Path(__file__).resolve().parents[3] / "config" / "reranker_weights.json",
        Path("config/reranker_weights.json"),
        Path("../config/reranker_weights.json"),
    ]
    for cp in config_paths:
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "learned_weights" in data and "weights" in data["learned_weights"]:
                    return normalize_weights(data["learned_weights"]["weights"])
            except Exception:
                pass

    # Fallback to fixed baseline if learned weights not found
    return dict(FIXED_BASELINE_WEIGHTS)


# ---------------------------------------------------------------------------
# Hard Legal Eligibility Constraint
# ---------------------------------------------------------------------------

def is_case_eligible(
    hc: Optional[models.HistoricalCase],
    excluded_case_ids: Optional[Set[str]] = None,
) -> bool:
    """
    Hard Legal Constraint:
        Eligible(d) = 0 if d is overruled/ineligible, 1 otherwise.
        FinalCandidates = {d : Eligible(d) = 1} followed by reranking.

    Note: In the absence of certified overruled annotations in the current corpus,
    all historical cases evaluate to True (filter structurally active, but non-excluding
    until reliable overruled status metadata is ingested).
    """
    if not hc:
        return False

    if excluded_case_ids and hc.id in excluded_case_ids:
        return False

    # Check explicit model attributes if present in schema
    if hasattr(hc, "is_overruled") and getattr(hc, "is_overruled"):
        return False

    # Check reliable metadata if present
    if hasattr(hc, "status") and getattr(hc, "status") == "overruled":
        return False

    return True


# ---------------------------------------------------------------------------
# Individual Feature Extractors
# ---------------------------------------------------------------------------

def compute_statutory_overlap_jaccard(
    query_provisions: Set[str],
    candidate_provisions: Set[str],
) -> float:
    """
    Statutory Overlap using Jaccard Similarity (Section 4.3):
        J(S_q, S_d) = |S_q ∩ S_d| / |S_q ∪ S_d|

    Handles empty set explicitly to avoid division by zero.
    """
    if not query_provisions or not candidate_provisions:
        return 0.0

    union_set = query_provisions | candidate_provisions
    if not union_set:
        return 0.0

    inter_set = query_provisions & candidate_provisions
    return round(len(inter_set) / len(union_set), 4)


def compute_issue_alignment_overlap(
    query_issues_tokens: Set[str],
    candidate_summary_tokens: Set[str],
) -> float:
    """
    Legal Issue Alignment using Overlap Coefficient (Section 4.4):
        Overlap(A, B) = |A ∩ B| / min(|A|, |B|)

    Handles empty set explicitly to avoid division by zero.
    """
    if not query_issues_tokens or not candidate_summary_tokens:
        return 0.0

    min_size = min(len(query_issues_tokens), len(candidate_summary_tokens))
    if min_size == 0:
        return 0.0

    inter_size = len(query_issues_tokens & candidate_summary_tokens)
    return round(min(1.0, inter_size / min_size), 4)


def compute_court_hierarchy_score(
    court_str: str,
    hierarchy_map: Optional[Dict[str, float]] = None,
) -> float:
    """
    Court Hierarchy Score (Section 4.6):
        Engineering encoding of Article 141 (Supreme Court binding authority).
    """
    h_map = hierarchy_map or DEFAULT_COURT_HIERARCHY
    c_lower = (court_str or "").lower()

    if "supreme court" in c_lower:
        return h_map.get("supreme court", 1.0)
    elif "high court" in c_lower:
        return h_map.get("high court", 0.75)
    elif "district" in c_lower or "sessions" in c_lower:
        return h_map.get("district court", 0.50)
    elif "tribunal" in c_lower:
        return h_map.get("tribunal", 0.40)
    return h_map.get("default", 0.50)


def extract_candidate_features(
    cand: RetrievedChunk,
    hc: models.HistoricalCase,
    target_provisions: Set[str],
    target_issues_tokens: Set[str],
    graph: Any,
    court_hierarchy_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Extracts and normalizes the six mathematical features for candidate (q, d):
        s_hat_j in [0.0, 1.0] for all j in {1..6}.
    """
    # 1. Semantic similarity (clamped cosine similarity)
    s_sem = max(0.0, min(1.0, float(cand.semantic_score)))

    # 2. Lexical similarity (BM25+ normalized token score)
    s_lex = max(0.0, min(1.0, float(cand.keyword_score)))

    # 3. Statutory overlap (Jaccard similarity on provisions)
    cand_provisions = set()
    if isinstance(hc.legal_sections, list):
        for sec in hc.legal_sections:
            if sec:
                cand_provisions.add(str(sec).strip().lower())
    # Also parse provisions mentioned in the candidate chunk text
    chunk_tokens = _tokenize_text(cand.chunk.text)
    for prov in target_provisions:
        prov_tokens = _tokenize_text(prov)
        if prov_tokens and prov_tokens.issubset(chunk_tokens):
            cand_provisions.add(prov)

    s_stat = compute_statutory_overlap_jaccard(target_provisions, cand_provisions)

    # 4. Issue alignment (Overlap coefficient)
    cand_summary_tokens = _tokenize_text(hc.summary or "") | _tokenize_text(cand.chunk.text[:500])
    s_issue = compute_issue_alignment_overlap(target_issues_tokens, cand_summary_tokens)

    # 5. Citation authority (Logarithmic in-degree dampening)
    s_cit = graph.get_authority_score(hc.id, mode="log")

    # 6. Court hierarchy (Article 141 engineering mapping)
    s_court = compute_court_hierarchy_score(hc.court or "", court_hierarchy_map)

    return {
        "semantic": s_sem,
        "lexical": s_lex,
        "statute": s_stat,
        "issue": s_issue,
        "citation": s_cit,
        "court": s_court,
    }


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RerankedCandidate:
    chunk: models.Chunk
    historical_case: models.HistoricalCase
    total_score: float
    semantic_score: float
    lexical_score: float
    statute_overlap_score: float
    issue_alignment_score: float
    citation_authority_score: float
    court_hierarchy_score: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "historical_case_id": self.historical_case.id,
            "case_name": self.historical_case.case_name,
            "court": self.historical_case.court,
            "year": self.historical_case.year,
            "citation": self.historical_case.citation,
            "total_score": round(self.total_score, 4),
            "score_breakdown": {
                "semantic": round(self.semantic_score, 4),
                "lexical": round(self.lexical_score, 4),
                "statute_overlap": round(self.statute_overlap_score, 4),
                "issue_alignment": round(self.issue_alignment_score, 4),
                "citation_authority": round(self.citation_authority_score, 4),
                "court_weight": round(self.court_hierarchy_score, 4),
            },
            "explanation": self.explanation,
            "excerpt": self.chunk.text[:300],
        }


# ---------------------------------------------------------------------------
# Main Reranking Function
# ---------------------------------------------------------------------------

def rerank_candidates(
    db: Session,
    query_case: models.Case,
    candidates: List[RetrievedChunk],
    top_k: int = 6,
    weights: Optional[Dict[str, float]] = None,
    mode: Optional[str] = None,
    excluded_case_ids: Optional[Set[str]] = None,
) -> List[RerankedCandidate]:
    """
    Reranks candidate chunks using the Constrained Additive Utility Model:
        S(q, d) = sum_j w_j * s_hat_j(q, d)
        subject to: w_j >= 0, sum w_j = 1.0, s_hat_j in [0, 1].

    Parameters:
    -----------
    db                : SQLAlchemy database session
    query_case        : Live user case with facts, issues, provisions
    candidates        : Pool of retrieved chunks from Stage 1 (Hybrid retrieval)
    top_k             : Number of top-ranked candidates to return
    weights           : Optional explicit weight dictionary
    mode              : 'fixed' (MCDA baseline) or 'learned' (validation-optimized)
    excluded_case_ids : Set of case IDs to hard-filter (e.g., overruled cases)
    """
    if not candidates:
        return []

    w = get_reranker_weights(mode=mode, custom_weights=weights)
    graph = get_citation_graph(db)

    # 1. Parse target case provisions
    target_provisions: Set[str] = set()
    if isinstance(query_case.provisions, list):
        for p in query_case.provisions:
            if isinstance(p, dict):
                act = p.get("act", "")
                sec = p.get("section", "")
                clean_prov = f"{sec} {act}".strip().lower()
                if clean_prov:
                    target_provisions.add(clean_prov)
            elif isinstance(p, str) and p.strip():
                target_provisions.add(p.strip().lower())

    # 2. Parse target case issue tokens
    target_issues_tokens = _tokenize_text(query_case.issues or "")

    reranked: List[RerankedCandidate] = []
    cand_hc_ids = list({cand.chunk.historical_case_id for cand in candidates if cand.chunk.historical_case_id})
    case_cache: Dict[str, Optional[models.HistoricalCase]] = {}
    if cand_hc_ids:
        for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(cand_hc_ids)).all():
            case_cache[hc.id] = hc

    for cand in candidates:
        hc_id = cand.chunk.historical_case_id
        hc = case_cache.get(hc_id)
        if not hc:
            continue

        # Apply Hard Legal Eligibility Constraint
        if not is_case_eligible(hc, excluded_case_ids=excluded_case_ids):
            continue

        # Extract normalized features in [0.0, 1.0]
        feats = extract_candidate_features(
            cand=cand,
            hc=hc,
            target_provisions=target_provisions,
            target_issues_tokens=target_issues_tokens,
            graph=graph,
        )

        # Compute Constrained Additive Utility Score: S(q, d) = sum_j w_j * s_hat_j
        total = (
            w.get("semantic", 0.0) * feats["semantic"]
            + w.get("lexical", 0.0) * feats["lexical"]
            + w.get("statute", 0.0) * feats["statute"]
            + w.get("issue", 0.0) * feats["issue"]
            + w.get("citation", 0.0) * feats["citation"]
            + w.get("court", 0.0) * feats["court"]
        )

        explanation_parts = []
        if feats["semantic"] > 0.4:
            explanation_parts.append("strong semantic narrative similarity")
        if feats["lexical"] > 0.3:
            explanation_parts.append("high keyword overlap")
        if feats["statute"] > 0.1:
            explanation_parts.append("statutory provision overlap")
        if feats["issue"] > 0.2:
            explanation_parts.append("contested legal issue alignment")
        if feats["citation"] > 0.2:
            explanation_parts.append("established precedential citation authority")
        if feats["court"] >= 0.9:
            explanation_parts.append("binding Supreme Court precedent (Art. 141)")

        explanation = (
            "Retrieved due to " + ", ".join(explanation_parts) + "."
            if explanation_parts
            else "Retrieved as relevant context based on multi-signal matching."
        )

        reranked.append(RerankedCandidate(
            chunk=cand.chunk,
            historical_case=hc,
            total_score=total,
            semantic_score=feats["semantic"],
            lexical_score=feats["lexical"],
            statute_overlap_score=feats["statute"],
            issue_alignment_score=feats["issue"],
            citation_authority_score=feats["citation"],
            court_hierarchy_score=feats["court"],
            explanation=explanation,
        ))

    # Sort descending by total score; break ties deterministically by semantic score then chunk ID
    reranked.sort(key=lambda x: (x.total_score, x.semantic_score, str(x.chunk.id)), reverse=True)
    return reranked[:top_k]
