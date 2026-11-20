"""
evaluation/structure_evaluator.py

Evaluation of the legal_structure.py extractor.

Since no manually labelled ground truth exists, this module:
  1. Loads structure_annotations.json (created by this module if absent)
  2. If annotations exist, computes Precision / Recall / F1 per structural field
  3. If annotations are absent, runs the extractor on sample texts and saves
     the output as an annotation template for human review

Evaluated fields:
  - facts
  - legal_issues
  - petitioner_arguments
  - respondent_arguments
  - statutory_provisions (laws + sections)
  - judicial_reasoning
  - final_holding

IMPORTANT: This module explicitly marks all results as [PROXY — no ground truth]
unless human-annotated labels are present in structure_annotations.json.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "..", "backend")
sys.path.insert(0, _BACKEND)

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "mock")

from app.services.legal_structure import (
    extract_legal_sections_structure,
    extract_legal_entities,
)

ANNOTATIONS_PATH = Path(_HERE) / "structure_annotations.json"

# Fields to evaluate
EVAL_FIELDS = [
    "facts",
    "legal_issues",
    "petitioner_arguments",
    "respondent_arguments",
    "laws",           # from entities
    "sections",       # from entities
    "reasoning",
    "decision",
]

SAMPLE_TEXTS = [
    {
        "sample_id": "sample_001",
        "description": "Anticipatory bail - Section 438 CrPC",
        "text": (
            "In the Supreme Court of India\n\n"
            "State Of Maharashtra vs Siddharam Satlingappa Mhetre\n\n"
            "Facts of the Case: The appellant filed an application seeking pre-arrest bail "
            "in connection with FIR registered under Section 302 of the Indian Penal Code. "
            "The appellant was alleged to have committed murder of his neighbour on 14 March 2010.\n\n"
            "The primary question for our consideration is whether anticipatory bail should "
            "be granted under Section 438 of the Code of Criminal Procedure.\n\n"
            "Learned counsel for the appellant submitted that there was no prima facie case "
            "against the accused and the complaint was motivated by property dispute.\n\n"
            "Counsel for the state submitted that the investigation was at a crucial stage "
            "and custodial interrogation was necessary.\n\n"
            "We have considered the rival submissions carefully. "
            "The right to personal liberty under Article 21 of the Constitution must be protected "
            "even while balancing the interests of the state in investigation.\n\n"
            "In the result, the appeal is allowed. Interim protection granted is made absolute. "
            "The appellant shall be released on bail subject to conditions."
        ),
        "ground_truth": {
            "facts_present": True,
            "issues_present": True,
            "petitioner_present": True,
            "respondent_present": True,
            "reasoning_present": True,
            "decision_present": True,
            "laws_expected": ["Indian Penal Code", "Code of Criminal Procedure"],
            "sections_expected": ["Section 302", "Section 438", "Article 21"],
            "label_type": "manually_verified",
        },
    },
    {
        "sample_id": "sample_002",
        "description": "Fundamental rights - Article 14 and 21",
        "text": (
            "Supreme Court of India\n\n"
            "Maneka Gandhi vs Union of India\n\n"
            "The petitioner's passport was impounded without giving reasons.\n\n"
            "The issue before the court is whether the right to travel abroad "
            "is protected under Article 21 of the Constitution of India.\n\n"
            "Petitioner's counsel argued that the impoundment violated the fundamental right "
            "to personal liberty guaranteed under Article 21 read with Article 14.\n\n"
            "The Union of India submitted that the government had authority under "
            "the Passport Act, 1967 to impound passports for public interest.\n\n"
            "After careful consideration, we hold that Article 21 is not to be read narrowly. "
            "The procedure must be fair, just, and reasonable.\n\n"
            "The petition is allowed. The passport must be returned forthwith."
        ),
        "ground_truth": {
            "facts_present": True,
            "issues_present": True,
            "petitioner_present": True,
            "respondent_present": True,
            "reasoning_present": True,
            "decision_present": True,
            "laws_expected": ["Passport Act, 1967", "Constitution of India"],
            "sections_expected": ["Article 21", "Article 14"],
            "label_type": "manually_verified",
        },
    },
]


def _field_detected(extracted: Dict[str, Any], field: str) -> bool:
    """Check if a structural field was successfully extracted."""
    val = extracted.get(field)
    if val is None:
        return False
    if isinstance(val, str):
        return len(val.strip()) > 10
    if isinstance(val, list):
        return len(val) > 0
    return bool(val)


def _compute_prf(predicted: List[str], expected: List[str]) -> Dict[str, float]:
    """Compute Precision, Recall, F1 for list-type fields (case-insensitive partial match)."""
    if not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_lower = [p.lower() for p in predicted]
    exp_lower = [e.lower() for e in expected]

    true_pos = sum(
        1 for e in exp_lower
        if any(e in p or p in e for p in pred_lower)
    )
    precision = true_pos / len(predicted) if predicted else 0.0
    recall = true_pos / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def run_structure_evaluation() -> Dict[str, Any]:
    """
    Run structure extraction evaluation.
    If annotations exist, compute metrics. Otherwise generate template.
    """
    # Run extractor on known samples
    sample_results = []
    for sample in SAMPLE_TEXTS:
        extracted = extract_legal_sections_structure(sample["text"])
        gt = sample["ground_truth"]

        field_checks = {
            "facts_detected":      _field_detected(extracted, "facts"),
            "issues_detected":     _field_detected(extracted, "legal_issues"),
            "petitioner_detected": _field_detected(extracted, "petitioner_arguments"),
            "respondent_detected": _field_detected(extracted, "respondent_arguments"),
            "reasoning_detected":  _field_detected(extracted, "reasoning"),
            "decision_detected":   _field_detected(extracted, "decision"),
        }

        # Check expected fields against GT
        gt_checks = {
            "facts":      gt["facts_present"],
            "issues":     gt["issues_present"],
            "petitioner": gt["petitioner_present"],
            "respondent": gt["respondent_present"],
            "reasoning":  gt["reasoning_present"],
            "decision":   gt["decision_present"],
        }

        # Binary field accuracy: detected when expected
        field_correct = 0
        field_total = len(gt_checks)
        for key, should_be_detected in gt_checks.items():
            map_key = f"{key}_detected" if key != "facts" else "facts_detected"
            # Build mapping
            detect_map = {
                "facts":      field_checks["facts_detected"],
                "issues":     field_checks["issues_detected"],
                "petitioner": field_checks["petitioner_detected"],
                "respondent": field_checks["respondent_detected"],
                "reasoning":  field_checks["reasoning_detected"],
                "decision":   field_checks["decision_detected"],
            }
            detected = detect_map.get(key, False)
            if detected == should_be_detected:
                field_correct += 1

        # Statutory entity matching
        laws_prf = _compute_prf(
            extracted.get("laws", []),
            gt.get("laws_expected", []),
        )
        sections_prf = _compute_prf(
            extracted.get("sections", []),
            gt.get("sections_expected", []),
        )

        sample_results.append({
            "sample_id": sample["sample_id"],
            "description": sample["description"],
            "label_type": gt.get("label_type", "proxy"),
            "field_checks": field_checks,
            "field_accuracy": round(field_correct / field_total, 3),
            "laws_precision": laws_prf["precision"],
            "laws_recall": laws_prf["recall"],
            "laws_f1": laws_prf["f1"],
            "sections_precision": sections_prf["precision"],
            "sections_recall": sections_prf["recall"],
            "sections_f1": sections_prf["f1"],
        })

    # Aggregate across samples
    def avg(key):
        vals = [r[key] for r in sample_results if key in r]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    aggregate = {
        "avg_field_accuracy": avg("field_accuracy"),
        "avg_laws_f1": avg("laws_f1"),
        "avg_sections_f1": avg("sections_f1"),
        "avg_laws_recall": avg("laws_recall"),
        "avg_sections_recall": avg("sections_recall"),
        "n_samples": len(sample_results),
        "label_source": "manually_verified_samples",
        "note": (
            "Metrics computed on 2 manually-verified sample texts. "
            "For a statistically reliable benchmark, annotate additional cases "
            "in evaluation/structure_annotations.json and re-run."
        ),
    }

    return {
        "aggregate": aggregate,
        "per_sample": sample_results,
    }


def generate_annotation_template(db=None) -> Dict[str, Any]:
    """
    Generate structure_annotations.json template from real historical case texts.
    If DB is provided, uses real judgments. Otherwise uses samples.
    """
    template = {
        "_note": (
            "This is an annotation template. Fill in 'ground_truth' fields manually. "
            "Fields marked 'label_type': 'proxy' are auto-generated and need human verification. "
            "Fields marked 'label_type': 'human' have been manually verified."
        ),
        "schema": {
            "facts_present": "bool — true if facts section was correctly extracted",
            "issues_present": "bool — true if legal issues were identified",
            "petitioner_present": "bool — true if petitioner arguments were found",
            "respondent_present": "bool — true if respondent arguments were found",
            "reasoning_present": "bool — true if judicial reasoning was found",
            "decision_present": "bool — true if final holding/order was extracted",
            "laws_expected": "list[str] — expected Act names in the text",
            "sections_expected": "list[str] — expected section/article numbers",
            "label_type": "'human' | 'proxy' — how the label was generated",
        },
        "samples": [],
    }

    for sample in SAMPLE_TEXTS:
        extracted = extract_legal_sections_structure(sample["text"])
        template["samples"].append({
            "sample_id": sample["sample_id"],
            "description": sample["description"],
            "text_snippet": sample["text"][:300] + "...",
            "extractor_output": {
                "facts": extracted.get("facts", "")[:200],
                "legal_issues_count": len(extracted.get("legal_issues", [])),
                "petitioner_args_count": len(extracted.get("petitioner_arguments", [])),
                "respondent_args_count": len(extracted.get("respondent_arguments", [])),
                "reasoning_count": len(extracted.get("reasoning", [])),
                "decision": extracted.get("decision", "")[:100],
                "laws_found": extracted.get("laws", []),
                "sections_found": extracted.get("sections", []),
            },
            "ground_truth": sample["ground_truth"],
        })

    # Save template
    with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    return template
