import os, re, json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Union
from sqlalchemy.orm import Session
from .. import models
from .retrieval import RetrievedChunk
from .citation_graph import get_citation_graph

FIXED_BASELINE_WEIGHTS: Dict[str, float] = {
    "semantic": 0.35, "lexical": 0.20, "statute": 0.20,
    "issue": 0.10, "citation": 0.08, "court": 0.07,
}
DEFAULT_COURT_HIERARCHY: Dict[str, float] = {
    "supreme court": 1.0, "high court": 0.75, "district court": 0.50,
    "tribunal": 0.40, "default": 0.50,
}
STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were",
    "for", "by", "with", "that", "this", "it", "as", "at", "be", "are", "from",
    "has", "have", "had", "which", "who", "whom", "not", "but", "his", "her",
    "their", "its", "he", "she", "they", "will", "shall", "may", "such",
    "under", "case", "court", "vs", "versus", "appeal", "appellant", "respondent"
}

def _tokenize_text(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()) if w not in STOPWORDS}

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in w.items()}
    tot = sum(cleaned.values())
    return {k: round(v / tot, 6) for k, v in cleaned.items()} if tot > 0 else {k: 1.0 / len(cleaned) for k in cleaned}

def get_reranker_weights(mode: Optional[str] = None, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if custom_weights:
        return normalize_weights(custom_weights)
    if (mode or os.getenv("RERANKER_WEIGHT_MODE") or "fixed").lower() == "fixed":
        return dict(FIXED_BASELINE_WEIGHTS)
    for cp in [Path(__file__).resolve().parents[3] / "config" / "reranker_weights.json", Path("config/reranker_weights.json")]:
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "learned_weights" in data and "weights" in data["learned_weights"]:
                    return normalize_weights(data["learned_weights"]["weights"])
            except Exception:
                pass
    return dict(FIXED_BASELINE_WEIGHTS)

def is_case_eligible(historical_case: Any, excluded_case_ids: Optional[Set[str]] = None) -> bool:
    if historical_case is None:
        return False
    cid = getattr(historical_case, "id", None)
    if excluded_case_ids and cid in excluded_case_ids:
        return False
    if getattr(historical_case, "is_overruled", False):
        return False
    if getattr(historical_case, "status", None) == "overruled":
        return False
    sn = (getattr(historical_case, "source_note", "") or "").lower()
    return not ("overruled" in sn or "bad law" in sn)

def compute_statutory_overlap_jaccard(target_provisions: Set[str], candidate_case: Union[Set[str], Any]) -> float:
    if not target_provisions or not candidate_case:
        return 0.0
    cand_provs: Set[str] = set()
    if isinstance(candidate_case, (set, list, tuple)):
        cand_provs = {str(x).strip().lower() for x in candidate_case if str(x).strip()}
    else:
        for coll in [getattr(candidate_case, "legal_sections", None), getattr(candidate_case, "laws", None)]:
            if isinstance(coll, list):
                for item in coll:
                    val = (item.get("section", "") if isinstance(item, dict) else str(item)).strip().lower()
                    if val: cand_provs.add(val)
    union = target_provisions | cand_provs
    return round(len(target_provisions & cand_provs) / len(union), 4) if union else 0.0

def compute_issue_alignment_overlap(target_issue_tokens: Set[str], candidate_case: Union[Set[str], Any]) -> float:
    if not target_issue_tokens or not candidate_case:
        return 0.0
    if isinstance(candidate_case, (set, list, tuple)):
        cand_tokens = {str(x).strip().lower() for x in candidate_case if str(x).strip()}
    else:
        cand_tokens = _tokenize_text(f"{getattr(candidate_case, 'case_name', '')} {getattr(candidate_case, 'summary', '')} {getattr(candidate_case, 'outcome', '')}")
    min_len = min(len(target_issue_tokens), len(cand_tokens))
    return round(len(target_issue_tokens & cand_tokens) / min_len, 4) if min_len > 0 else 0.0

def compute_court_hierarchy_score(court_name: Optional[str], hierarchy_map: Optional[Dict[str, float]] = None) -> float:
    h = hierarchy_map or DEFAULT_COURT_HIERARCHY
    c_clean = (court_name or "").strip().lower()
    if not c_clean: return h["default"]
    for k, v in h.items():
        if k in c_clean: return v
    return h["default"]

def extract_candidate_features(
    cand: RetrievedChunk,
    hc: models.HistoricalCase,
    target_provisions: Set[str],
    target_issues_tokens: Set[str],
    graph: Any,
) -> Dict[str, float]:
    return {
        "semantic": round(max(0.0, min(1.0, float(getattr(cand, "semantic_score", 0.0)))), 4),
        "lexical": round(max(0.0, min(1.0, float(getattr(cand, "keyword_score", 0.0)))), 4),
        "statute": compute_statutory_overlap_jaccard(target_provisions, hc),
        "issue": compute_issue_alignment_overlap(target_issues_tokens, hc),
        "citation": graph.get_authority_score(hc.id, mode="log"),
        "court": compute_court_hierarchy_score(hc.court),
    }

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
            "matching_excerpt": (self.chunk.text or "")[:400],
        }

def rerank_candidates(
    db: Session,
    query_case: models.Case,
    candidates: List[RetrievedChunk],
    top_k: int = 6,
    weights: Optional[Dict[str, float]] = None,
    mode: Optional[str] = None,
    excluded_case_ids: Optional[Set[str]] = None,
) -> List[RerankedCandidate]:
    if not candidates:
        return []
    w = get_reranker_weights(mode=mode, custom_weights=weights)
    graph = get_citation_graph(db)

    target_provisions: Set[str] = set()
    if isinstance(query_case.provisions, list):
        for p in query_case.provisions:
            val = (p.get("section", "") + " " + p.get("act", "") if isinstance(p, dict) else str(p)).strip().lower()
            if val: target_provisions.add(val)

    target_issues_tokens = _tokenize_text(query_case.issues or "")
    cand_hc_ids = list({c.chunk.historical_case_id for c in candidates if c.chunk.historical_case_id})
    case_cache = {hc.id: hc for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(cand_hc_ids)).all()} if cand_hc_ids else {}

    reranked: List[RerankedCandidate] = []
    for cand in candidates:
        hc = case_cache.get(cand.chunk.historical_case_id)
        if not hc or not is_case_eligible(hc, excluded_case_ids=excluded_case_ids):
            continue

        f = extract_candidate_features(cand, hc, target_provisions, target_issues_tokens, graph)
        tot = (w.get("semantic", 0.0) * f["semantic"] + w.get("lexical", 0.0) * f["lexical"]
               + w.get("statute", 0.0) * f["statute"] + w.get("issue", 0.0) * f["issue"]
               + w.get("citation", 0.0) * f["citation"] + w.get("court", 0.0) * f["court"])

        expl_parts = []
        if f["semantic"] > 0.4: expl_parts.append("strong semantic narrative similarity")
        if f["lexical"] > 0.3: expl_parts.append("high keyword overlap")
        if f["statute"] > 0.1: expl_parts.append("statutory provision overlap")
        if f["issue"] > 0.2: expl_parts.append("contested legal issue alignment")
        if f["citation"] > 0.2: expl_parts.append("established precedential citation authority")
        if f["court"] >= 0.9: expl_parts.append("binding Supreme Court precedent (Art. 141)")
        expl = f"Retrieved due to {', '.join(expl_parts)}." if expl_parts else "Retrieved as relevant context based on multi-signal matching."

        reranked.append(RerankedCandidate(
            chunk=cand.chunk, historical_case=hc, total_score=tot,
            semantic_score=f["semantic"], lexical_score=f["lexical"],
            statute_overlap_score=f["statute"], issue_alignment_score=f["issue"],
            citation_authority_score=f["citation"], court_hierarchy_score=f["court"],
            explanation=expl,
        ))

    reranked.sort(key=lambda x: (x.total_score, x.semantic_score, str(x.chunk.id)), reverse=True)
    return reranked[:top_k]
