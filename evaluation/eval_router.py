"""
evaluation/eval_router.py

FastAPI router exposing evaluation results as REST endpoints.
Mount this into the existing Nya(AI) FastAPI app to add an Evaluation tab.

Endpoints:
  GET /evaluation/summary     -> DB stats + evaluation dataset info
  GET /evaluation/retrieval   -> Retrieval baseline metrics
  GET /evaluation/ablation    -> Ablation study results
  GET /evaluation/provenance  -> Citation provenance validation
  GET /evaluation/performance -> Latency benchmarks
  GET /evaluation/structure   -> Structure extraction metrics
  GET /evaluation/run-quick   -> Trigger a quick evaluation run (async)
"""
import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _load_result(filename: str) -> Optional[Dict[str, Any]]:
    """Load a result JSON file if it exists."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def _result_or_404(filename: str, label: str) -> Dict[str, Any]:
    data = _load_result(filename)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{label} results not found. "
                "Run: python evaluation/run_evaluation.py first."
            ),
        )
    return data


@router.get("/summary")
def evaluation_summary():
    """
    Summary of the evaluation dataset and corpus statistics.
    """
    # Load queries
    queries_path = _HERE / "queries.json"
    n_queries = 0
    if queries_path.exists():
        with open(queries_path, encoding="utf-8") as f:
            q_data = json.load(f)
        n_queries = len(q_data.get("queries", []))

    # Load ground truth
    gt_path = _HERE / "ground_truth.json"
    gt_populated = False
    gt_label_confidence = "not generated"
    n_relevant_queries = 0
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_label_confidence = gt_data.get("_label_confidence", "unknown")
        gt_entries = gt_data.get("ground_truth", [])
        n_relevant_queries = sum(1 for e in gt_entries if e.get("relevant_cases"))
        gt_populated = n_relevant_queries > 0

    # Available result files
    result_files = {
        "retrieval_metrics": (RESULTS_DIR / "retrieval_metrics.json").exists(),
        "ablation_results": (RESULTS_DIR / "ablation_results.json").exists(),
        "weight_sensitivity": (RESULTS_DIR / "weight_sensitivity.json").exists(),
        "structure_metrics": (RESULTS_DIR / "structure_metrics.json").exists(),
        "provenance_validation": (RESULTS_DIR / "provenance_validation.json").exists(),
        "performance": (RESULTS_DIR / "performance.json").exists(),
    }

    # Figures
    figures = []
    if (RESULTS_DIR / "figures").exists():
        figures = [f.name for f in (RESULTS_DIR / "figures").iterdir()
                   if f.suffix == ".png"]

    return {
        "evaluation_framework": "Nya(AI) Mid-Sem Evaluation Framework",
        "label_warning": "All retrieval metrics use PROXY labels — not human-annotated",
        "dataset": {
            "n_queries": n_queries,
            "ground_truth_populated": gt_populated,
            "n_queries_with_relevant_cases": n_relevant_queries,
            "label_confidence": gt_label_confidence,
        },
        "available_results": result_files,
        "available_figures": figures,
        "how_to_run": "python evaluation/run_evaluation.py [--quick]",
    }


@router.get("/retrieval")
def evaluation_retrieval():
    """
    Retrieval baseline comparison metrics.
    Includes BM25, Dense, Hybrid, and Full Reranker results.
    """
    return _result_or_404("retrieval_metrics.json", "Retrieval metrics")


@router.get("/ablation")
def evaluation_ablation():
    """
    Ablation study results (A0–A5 signal configurations).
    """
    result = _result_or_404("ablation_results.json", "Ablation results")
    report_path = RESULTS_DIR / "ABLATION_REPORT.md"
    if report_path.exists():
        result["report_available"] = True
        result["report_path"] = str(report_path)
    return result


@router.get("/weight-sensitivity")
def evaluation_weight_sensitivity():
    """
    Weight sensitivity analysis across 5 weight configurations.
    """
    result = _result_or_404("weight_sensitivity.json", "Weight sensitivity results")
    ws_report = RESULTS_DIR / "WEIGHT_SENSITIVITY_REPORT.md"
    if ws_report.exists():
        result["report_available"] = True
    return result


@router.get("/structure")
def evaluation_structure():
    """
    Legal structure extraction evaluation metrics.
    """
    return _result_or_404("structure_metrics.json", "Structure extraction metrics")


@router.get("/provenance")
def evaluation_provenance():
    """
    Citation provenance / zero-hallucination compliance results.
    """
    return _result_or_404("provenance_validation.json", "Provenance validation results")


@router.get("/performance")
def evaluation_performance():
    """
    Pipeline latency benchmark results (mean, median, P95 per stage).
    """
    return _result_or_404("performance.json", "Performance benchmark results")


@router.get("/report")
def evaluation_report():
    """
    Returns the final mid-sem evaluation report as text.
    """
    report_path = _HERE / "MIDSEM_EVALUATION_REPORT.md"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not generated yet. Run: python evaluation/run_evaluation.py"
        )
    with open(report_path, encoding="utf-8") as f:
        content = f.read()
    return {"report_markdown": content}


@router.post("/run-quick")
def run_quick_evaluation(background_tasks: BackgroundTasks):
    """
    Trigger a quick evaluation run in the background.
    Results will be available via other /evaluation/* endpoints after completion.
    """
    def _run():
        _backend = str(_HERE.parent / "backend")
        if _backend not in sys.path:
            sys.path.insert(0, _backend)
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))

        os.environ.setdefault("EMBEDDING_PROVIDER", "local")
        os.environ.setdefault("LLM_PROVIDER", "mock")

        try:
            import subprocess
            subprocess.run(
                [sys.executable, str(_HERE / "run_evaluation.py"), "--quick", "--skip-perf"],
                cwd=str(_HERE.parent),
                timeout=600,
            )
        except Exception as e:
            print(f"[Eval API] Background evaluation failed: {e}")

    background_tasks.add_task(_run)
    return {
        "status": "started",
        "message": "Quick evaluation started in background. Check /evaluation/summary for progress.",
        "note": "Results will be available after ~2-5 minutes.",
    }
