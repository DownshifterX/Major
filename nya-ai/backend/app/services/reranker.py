"""Multi-Signal Legal Reranking Engine.

Evaluates and re-scores candidates from hybrid retrieval using legal signals:
1. Semantic similarity (embedding cosine distance)
2. Lexical match (BM25 token overlap)
3. Statutory / Section overlap (e.g. matching Section 302, Article 21)
4. Legal Issue / Fact pattern alignment
5. Citation authority & Precedent centrality
6. Court hierarchy alignment (Supreme Court > High Court)

Every score component is transparently calculated and exposed.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .. import models
from .retrieval import RetrievedChunk
from .citation_graph import get_citation_graph


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
            "total_score": round(self.total_score, 3),
            "score_breakdown": {
                "semantic": round(self.semantic_score, 3),
                "lexical": round(self.lexical_score, 3),
                "statute_overlap": round(self.statute_overlap_score, 3),
                "issue_alignment": round(self.issue_alignment_score, 3),
                "citation_authority": round(self.citation_authority_score, 3),
                "court_weight": round(self.court_hierarchy_score, 3),
            },
            "explanation": self.explanation,
            "excerpt": self.chunk.text[:300],
        }


def rerank_candidates(
    db: Session,
    query_case: models.Case,
    candidates: List[RetrievedChunk],
    top_k: int = 6,
    weights: Optional[Dict[str, float]] = None,
) -> List[RerankedCandidate]:
    """Reranks candidate chunks with configurable weights."""
    if not candidates:
        return []

    w = weights or {
        "semantic": 0.35,
        "lexical": 0.20,
        "statute": 0.20,
        "issue": 0.10,
        "citation": 0.08,
        "court": 0.07,
    }

    graph = get_citation_graph(db)

    # Extract target case provisions and issues for matching
    target_provisions = set()
    if isinstance(query_case.provisions, list):
        for p in query_case.provisions:
            if isinstance(p, dict):
                act = p.get("act", "")
                sec = p.get("section", "")
                target_provisions.add(f"{sec} {act}".strip().lower())
            elif isinstance(p, str):
                target_provisions.add(p.lower())

    target_issues_text = (query_case.issues or "").lower()

    reranked: List[RerankedCandidate] = []
    case_cache: Dict[str, models.HistoricalCase] = {}

    for cand in candidates:
        hc_id = cand.chunk.historical_case_id
        if hc_id not in case_cache:
            case_cache[hc_id] = db.query(models.HistoricalCase).get(hc_id)
        hc = case_cache[hc_id]
        if not hc:
            continue

        # 1. Statute Overlap Score
        statute_score = 0.0
        shared_statutes = []
        chunk_text_lower = cand.chunk.text.lower()
        for prov in target_provisions:
            if prov and prov in chunk_text_lower:
                statute_score += 0.5
                shared_statutes.append(prov)
        statute_score = min(1.0, statute_score)

        # 2. Issue alignment score
        issue_score = 0.0
        if target_issues_text and hc.summary:
            common_words = set(target_issues_text.split()) & set(hc.summary.lower().split())
            issue_score = min(1.0, len(common_words) / 10.0)

        # 3. Citation Authority Score
        auth_score = graph.get_authority_score(hc.id)
        # Normalize cites_count if present
        if hc.cited_by_count:
            auth_score = max(auth_score, min(1.0, hc.cited_by_count / 100.0))

        # 4. Court Hierarchy Score
        court_score = 0.5
        court_str = (hc.court or "").lower()
        if "supreme court" in court_str:
            court_score = 1.0
        elif "high court" in court_str:
            court_score = 0.75

        total = (
            w["semantic"] * cand.semantic_score
            + w["lexical"] * cand.keyword_score
            + w["statute"] * statute_score
            + w["issue"] * issue_score
            + w["citation"] * auth_score
            + w["court"] * court_score
        )

        explanation_parts = []
        if cand.semantic_score > 0.4:
            explanation_parts.append("strong semantic narrative similarity")
        if cand.keyword_score > 0.3:
            explanation_parts.append("high keyword overlap")
        if shared_statutes:
            explanation_parts.append(f"shared statutory provisions ({', '.join(shared_statutes[:2])})")
        if "supreme court" in court_str:
            explanation_parts.append("binding Supreme Court precedent")
        if auth_score > 0.2:
            explanation_parts.append("frequently cited authority")

        explanation = (
            "Retrieved due to " + ", ".join(explanation_parts) + "."
            if explanation_parts
            else "Retrieved as relevant context based on multi-signal matching."
        )

        reranked.append(RerankedCandidate(
            chunk=cand.chunk,
            historical_case=hc,
            total_score=total,
            semantic_score=cand.semantic_score,
            lexical_score=cand.keyword_score,
            statute_overlap_score=statute_score,
            issue_alignment_score=issue_score,
            citation_authority_score=auth_score,
            court_hierarchy_score=court_score,
            explanation=explanation,
        ))

    # Sort descending by total multi-signal score
    reranked.sort(key=lambda x: x.total_score, reverse=True)
    return reranked[:top_k]
