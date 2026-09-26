"""
evaluation/weight_sensitivity.py

Weight sensitivity analysis for the six-signal reranker.

Tests 5 controlled weight configurations while keeping sum(weights) = 1.0.
Does NOT perform random tuning — only evaluates pre-defined configurations.

Configurations:
  Config A: Current/default weights (0.35/0.20/0.20/0.10/0.08/0.07)
  Config B: Higher semantic weight (emphasizes narrative similarity)
  Config C: Higher statutory weight (emphasizes section matching)
  Config D: Higher citation authority (emphasizes judicial authority)
  Config E: Balanced weights (equal contribution)
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "..", "backend")
sys.path.insert(0, _BACKEND)

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "mock")

from sqlalchemy.orm import Session
from app import models
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import rerank_candidates


# All configurations must sum to 1.0
WEIGHT_CONFIGS = {
    "config_A_default": {
        "description": "Default weights (current Nya(AI) configuration)",
        "weights": {
            "semantic": 0.35,
            "lexical":  0.20,
            "statute":  0.20,
            "issue":    0.10,
            "citation": 0.08,
            "court":    0.07,
        },
    },
    "config_B_high_semantic": {
        "description": "Increased semantic weight — prioritizes narrative similarity",
        "weights": {
            "semantic": 0.55,
            "lexical":  0.15,
            "statute":  0.15,
            "issue":    0.07,
            "citation": 0.05,
            "court":    0.03,
        },
    },
    "config_C_high_statutory": {
        "description": "Increased statutory weight — prioritizes section/act overlap",
        "weights": {
            "semantic": 0.25,
            "lexical":  0.15,
            "statute":  0.40,
            "issue":    0.10,
            "citation": 0.06,
            "court":    0.04,
        },
    },
    "config_D_high_citation": {
        "description": "Increased citation authority weight — prioritizes cited precedents",
        "weights": {
            "semantic": 0.28,
            "lexical":  0.17,
            "statute":  0.17,
            "issue":    0.08,
            "citation": 0.25,
            "court":    0.05,
        },
    },
    "config_E_balanced": {
        "description": "Balanced equal weights — each signal contributes equally",
        "weights": {
            "semantic": 0.167,
            "lexical":  0.167,
            "statute":  0.167,
            "issue":    0.167,
            "citation": 0.166,
            "court":    0.166,
        },
    },
}


def validate_weights(weights: Dict[str, float]) -> bool:
    """Verify that weights sum to approximately 1.0."""
    total = sum(weights.values())
    return abs(total - 1.0) < 0.01


@dataclass
class WeightSensitivityResult:
    config_name: str
    description: str
    weights: Dict[str, float]
    query_id: str
    ranked_cases: List[str]
    latency_ms: float


def run_weight_config(
    db: Session,
    config_name: str,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> WeightSensitivityResult:
    """Run one weight configuration for one query."""
    cfg = WEIGHT_CONFIGS[config_name]
    weights = cfg["weights"]

    if not validate_weights(weights):
        raise ValueError(
            f"Config '{config_name}' weights sum to {sum(weights.values()):.4f}, expected ~1.0"
        )

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

    return WeightSensitivityResult(
        config_name=config_name,
        description=cfg["description"],
        weights=weights,
        query_id=query_id,
        ranked_cases=ranked_cases[:top_k],
        latency_ms=latency_ms,
    )


def run_all_weight_configs(
    db: Session,
    query_id: str,
    query_text: str,
    query_case: Optional[models.Case] = None,
    top_k: int = 15,
) -> Dict[str, WeightSensitivityResult]:
    """Run all 5 weight configurations for one query."""
    results = {}
    for config_name in WEIGHT_CONFIGS:
        results[config_name] = run_weight_config(
            db, config_name, query_id, query_text, query_case, top_k
        )
    return results
