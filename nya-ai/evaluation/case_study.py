"""
evaluation/case_study.py

End-to-end case study evaluation.

Runs the complete pipeline for one evaluation case and saves intermediate
outputs at every stage for inspection and reproducibility.

Pipeline:
  01_input.json          -> Case metadata
  02_extracted_structure.json -> Legal structure extraction output
  03_retrieval_results.json   -> Hybrid retrieval output
  04_reranking_results.json   -> Six-signal reranking output
  05_citation_graph.json      -> Citation graph subgraph
  06_analysis.json            -> Grounded LLM analysis output
  07_citation_verification.json -> Provenance check
  08_final_report.md          -> Human-readable summary

All outputs are written to evaluation/case_studies/<case_id>/
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "..", "backend")
sys.path.insert(0, _BACKEND)

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "mock")

from sqlalchemy.orm import Session
from app import models
from app.services.legal_structure import extract_legal_sections_structure
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import rerank_candidates
from app.services.citation_graph import get_citation_graph
from app.services.analysis_service import run_similar_cases
from provenance_validator import validate_analysis_provenance


CASE_STUDIES_DIR = Path(_HERE) / "case_studies"


def _save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def run_case_study(
    db: Session,
    case_title: str,
    case_text: str,
    case_facts: str,
    case_issues: str,
    case_type: str = "Criminal",
    provisions: Optional[list] = None,
    study_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full end-to-end pipeline for one case and save intermediate outputs.

    Returns dict of stage outputs.
    """
    if study_id is None:
        study_id = "case_" + str(abs(hash(case_title)))[:8]

    output_dir = CASE_STUDIES_DIR / study_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [case_study] Running study: {study_id}")
    print(f"  [case_study] Output dir: {output_dir}")

    # ---- 01 Input ----
    input_data = {
        "study_id": study_id,
        "case_title": case_title,
        "case_text_length": len(case_text),
        "case_facts_preview": case_facts[:300],
        "case_issues": case_issues,
        "case_type": case_type,
        "provisions": provisions or [],
    }
    _save_json(output_dir / "01_input.json", input_data)
    print("  [case_study] 01_input.json saved")

    # ---- 02 Legal Structure Extraction ----
    t0 = time.perf_counter()
    extracted = extract_legal_sections_structure(case_text)
    extraction_ms = (time.perf_counter() - t0) * 1000

    structure_data = {
        "study_id": study_id,
        "extraction_latency_ms": round(extraction_ms, 2),
        "facts_length": len(extracted.get("facts", "")),
        "issues_count": len(extracted.get("legal_issues", [])),
        "petitioner_args_count": len(extracted.get("petitioner_arguments", [])),
        "respondent_args_count": len(extracted.get("respondent_arguments", [])),
        "reasoning_count": len(extracted.get("reasoning", [])),
        "decision_length": len(extracted.get("decision", "")),
        "laws_found": extracted.get("laws", []),
        "sections_found": extracted.get("sections", []),
        "extracted": {k: v for k, v in extracted.items() if k != "entities"},
    }
    _save_json(output_dir / "02_extracted_structure.json", structure_data)
    print("  [case_study] 02_extracted_structure.json saved")

    # ---- Create DB Case record ----
    query_text = f"{case_facts}\n{case_issues}"
    db_case = models.Case(
        title=case_title[:120],
        facts=case_facts,
        issues=case_issues,
        case_type=case_type,
        provisions=provisions or [],
    )
    db.add(db_case)
    db.flush()

    # ---- 03 Retrieval Results ----
    t0 = time.perf_counter()
    retrieved = retrieve_historical_chunks(db, query_text, top_k=15, per_case_cap=3)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieval_data = {
        "study_id": study_id,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "n_chunks_retrieved": len(retrieved),
        "chunks": [
            {
                "chunk_id": r.chunk.id,
                "historical_case_id": r.chunk.historical_case_id,
                "semantic_score": round(r.semantic_score, 4),
                "keyword_score": round(r.keyword_score, 4),
                "hybrid_score": round(r.hybrid_score, 4),
                "text_preview": r.chunk.text[:200],
            }
            for r in retrieved
        ],
    }
    _save_json(output_dir / "03_retrieval_results.json", retrieval_data)
    print("  [case_study] 03_retrieval_results.json saved")

    # ---- 04 Reranking Results ----
    t0 = time.perf_counter()
    reranked = rerank_candidates(db, db_case, retrieved, top_k=8)
    reranking_ms = (time.perf_counter() - t0) * 1000

    reranking_data = {
        "study_id": study_id,
        "reranking_latency_ms": round(reranking_ms, 2),
        "n_reranked": len(reranked),
        "candidates": [r.to_dict() for r in reranked],
    }
    _save_json(output_dir / "04_reranking_results.json", reranking_data)
    print("  [case_study] 04_reranking_results.json saved")

    # ---- 05 Citation Graph ----
    if reranked:
        top_hc_id = reranked[0].historical_case.id
        t0 = time.perf_counter()
        graph = get_citation_graph(db)
        citation_subgraph = graph.get_case_citation_subgraph(top_hc_id, depth=1)
        graph_ms = (time.perf_counter() - t0) * 1000

        graph_data = {
            "study_id": study_id,
            "focus_case_id": top_hc_id,
            "focus_case_name": reranked[0].historical_case.case_name,
            "graph_query_latency_ms": round(graph_ms, 2),
            **citation_subgraph,
        }
    else:
        graph_data = {
            "study_id": study_id,
            "focus_case_id": None,
            "note": "No reranked candidates — citation graph not built",
        }
    _save_json(output_dir / "05_citation_graph.json", graph_data)
    print("  [case_study] 05_citation_graph.json saved")

    # ---- 06 Analysis (mock LLM) ----
    t0 = time.perf_counter()
    try:
        analysis_result = run_similar_cases(db, db_case)
        analysis_ok = True
    except Exception as e:
        analysis_result = {"error": str(e)}
        analysis_ok = False
    analysis_ms = (time.perf_counter() - t0) * 1000

    analysis_data = {
        "study_id": study_id,
        "analysis_latency_ms": round(analysis_ms, 2),
        "analysis_ok": analysis_ok,
        "analysis_id": analysis_result.get("analysis_id"),
        "result": analysis_result,
    }
    _save_json(output_dir / "06_analysis.json", analysis_data)
    print("  [case_study] 06_analysis.json saved")

    # ---- 07 Citation Verification ----
    if analysis_ok and analysis_result.get("analysis_id"):
        prov_report = validate_analysis_provenance(db, analysis_result["analysis_id"])
        citation_data = {
            "study_id": study_id,
            "analysis_id": analysis_result["analysis_id"],
            "total_citations": prov_report.total_citations,
            "fully_valid": prov_report.fully_valid,
            "chunk_missing": prov_report.chunk_missing,
            "case_missing": prov_report.case_missing,
            "offset_invalid": prov_report.offset_invalid,
            "excerpt_mismatch": prov_report.excerpt_mismatch,
            "provenance_pass_rate": prov_report.provenance_pass_rate,
            "note": (
                "Provenance measured against actual DB records. "
                "NOT from prompt instructions alone."
            ),
        }
    else:
        citation_data = {
            "study_id": study_id,
            "note": "Analysis did not complete — provenance not measurable",
        }
    _save_json(output_dir / "07_citation_verification.json", citation_data)
    print("  [case_study] 07_citation_verification.json saved")

    # ---- 08 Final Report ----
    top_case_name = reranked[0].historical_case.case_name if reranked else "N/A"
    top_score = round(reranked[0].total_score, 3) if reranked else 0.0
    top_explanation = reranked[0].explanation if reranked else "N/A"

    report_md = f"""# Case Study Report: {case_title}

**Study ID**: {study_id}  
**Case Type**: {case_type}

---

## Pipeline Stage Summary

| Stage | Result |
|---|---|
| Legal Structure Extraction | {extraction_ms:.1f} ms |
| Hybrid Retrieval | {retrieval_ms:.1f} ms, {len(retrieved)} chunks |
| Six-Signal Reranking | {reranking_ms:.1f} ms, {len(reranked)} candidates |
| Analysis (Mock LLM) | {analysis_ms:.1f} ms, ok={analysis_ok} |

---

## Top Retrieved Precedent

**Case**: {top_case_name}  
**Composite Score**: {top_score}  
**Explanation**: {top_explanation}

---

## Score Breakdown (Top Precedent)

{json.dumps(reranked[0].to_dict()["score_breakdown"] if reranked else {}, indent=2)}

---

## Laws & Sections Identified

**Acts**: {", ".join(extracted.get("laws", [])[:5]) or "None detected"}  
**Sections**: {", ".join(extracted.get("sections", [])[:8]) or "None detected"}

---

## Provenance

- Total Citations: {citation_data.get("total_citations", "N/A")}
- Fully Valid: {citation_data.get("fully_valid", "N/A")}
- Provenance Pass Rate: {citation_data.get("provenance_pass_rate", "N/A")}

---

*Generated by Nya(AI) Evaluation Framework*
"""

    with open(output_dir / "08_final_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("  [case_study] 08_final_report.md saved")

    # Rollback DB changes (don't pollute live DB with test cases)
    db.rollback()

    return {
        "study_id": study_id,
        "output_dir": str(output_dir),
        "stages_completed": 8,
        "top_case": top_case_name,
        "top_score": top_score,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "reranking_latency_ms": round(reranking_ms, 2),
        "analysis_latency_ms": round(analysis_ms, 2),
    }


# --- Default evaluation case ---
DEFAULT_CASE = {
    "case_title": "State vs Accused — Murder under Section 302 IPC",
    "case_text": (
        "In the High Court of Judicature at Bombay\n\n"
        "State of Maharashtra vs Ramesh Kumar\n\n"
        "Facts of the Case: On 5th April 2022, the deceased Suresh Patil was allegedly "
        "attacked with a sharp weapon at his residence in Pune district. The accused Ramesh Kumar "
        "was apprehended near the scene. FIR was registered under Section 302 of the Indian Penal Code.\n\n"
        "Issues for Determination: Whether the prosecution has established guilt beyond "
        "reasonable doubt. Whether eyewitness testimony corroborated by forensic evidence "
        "satisfies the standard under Section 302 IPC.\n\n"
        "Learned counsel for the accused submitted that the eyewitness was an interested "
        "witness being the brother of the deceased and his testimony should not be believed.\n\n"
        "The Public Prosecutor submitted that the forensic report confirms blood of the "
        "deceased on the accused's clothes and the eyewitness testimony was consistent "
        "and corroborated.\n\n"
        "We have carefully examined the evidence. The testimony of a related witness is "
        "not automatically discarded. If it is otherwise reliable and corroborated, "
        "it can form the basis of conviction. The forensic evidence here corroborates the "
        "eyewitness account.\n\n"
        "Conviction under Section 302 IPC is upheld. The appeal is dismissed."
    ),
    "case_facts": (
        "Accused allegedly stabbed deceased with sharp weapon. "
        "FIR under Section 302 IPC. Eyewitness is brother of deceased. "
        "Forensic report shows blood of deceased on accused's clothing."
    ),
    "case_issues": (
        "Whether guilt is established beyond reasonable doubt under Section 302 IPC. "
        "Reliability of related eyewitness. Evidentiary value of forensic evidence."
    ),
    "case_type": "Criminal",
    "provisions": [
        {"act": "Indian Penal Code", "section": "Section 302"},
    ],
}
