# Nya(AI) — Mid-Sem Evaluation Framework

## Overview

This directory contains the complete experimental evaluation framework for Nya(AI), an AI-Assisted Legal Intelligence & Decision-Support System for Indian Supreme Court and High Court jurisprudence.

## Quick Start

```bash
# Full evaluation (15-30 min)
python evaluation/run_evaluation.py

# Quick development test (~2-5 min)
python evaluation/run_evaluation.py --quick

# Skip slow steps
python evaluation/run_evaluation.py --skip-perf --skip-case-study
```

## Directory Structure

```
evaluation/
├── README.md                    <- This file
├── IMPLEMENTATION_AUDIT.md      <- Audit of existing system components
├── MIDSEM_EVALUATION_REPORT.md  <- Final auto-generated report
│
├── queries.json                 <- 20 evaluation queries
├── ground_truth.json            <- Auto-populated proxy relevance labels
├── structure_annotations.json   <- Structure extraction annotation template
│
├── run_evaluation.py            <- Master orchestrator
├── metrics.py                   <- IR metric functions (P@K, MRR, NDCG, etc.)
├── baselines.py                 <- 4 retrieval baselines (BM25, Dense, Hybrid, Full)
├── ablation.py                  <- 6 ablation configurations (A0-A5)
├── weight_sensitivity.py        <- 5 weight configurations (A-E)
├── structure_evaluator.py       <- Legal structure extraction F1 evaluation
├── provenance_validator.py      <- Citation chain verification
├── performance.py               <- Latency benchmarking
├── case_study.py                <- End-to-end pipeline case study
│
├── tests/
│   └── test_edge_cases.py       <- 12 edge case + metric correctness tests
│
├── results/
│   ├── retrieval_metrics.json
│   ├── retrieval_comparison.csv
│   ├── ablation_results.json
│   ├── ablation_results.csv
│   ├── ABLATION_REPORT.md
│   ├── weight_sensitivity.json
│   ├── weight_sensitivity.csv
│   ├── WEIGHT_SENSITIVITY_REPORT.md
│   ├── structure_metrics.json
│   ├── structure_metrics.csv
│   ├── provenance_validation.json
│   ├── performance.json
│   ├── performance.csv
│   └── figures/
│       ├── retrieval_comparison.png
│       ├── ablation_curve.png
│       └── latency_comparison.png
│
└── case_studies/
    └── <study_id>/
        ├── 01_input.json
        ├── 02_extracted_structure.json
        ├── 03_retrieval_results.json
        ├── 04_reranking_results.json
        ├── 05_citation_graph.json
        ├── 06_analysis.json
        ├── 07_citation_verification.json
        └── 08_final_report.md
```

## Evaluation Phases

| Phase | Description | Output |
|---|---|---|
| 1 | Database validation | Console stats |
| 2 | Proxy ground truth generation | `ground_truth.json` |
| 3+4 | 4 retrieval baselines + IR metrics | `retrieval_metrics.json`, `retrieval_comparison.csv` |
| 5 | Ablation study (A0-A5) | `ablation_results.json`, `ABLATION_REPORT.md` |
| 6 | Weight sensitivity (5 configs) | `weight_sensitivity.csv`, `WEIGHT_SENSITIVITY_REPORT.md` |
| 7 | Legal structure extraction eval | `structure_metrics.csv` |
| 8 | End-to-end case study | `case_studies/<id>/` |
| 10 | Provenance validation | `provenance_validation.json` |
| 11 | Performance benchmarks | `performance.csv` |
| 13 | Figures generation | `results/figures/` |
| 14 | Final report | `MIDSEM_EVALUATION_REPORT.md` |

## Ground Truth Methodology

> ⚠️ **IMPORTANT: All relevance labels are PROXY labels — not human-annotated.**

### How Proxy Labels Are Generated

A historical case is labeled **relevant** to a query if its `summary + laws + case_name` text contains 1+ expected keywords from `queries.json`. It is labeled **highly relevant** if it contains 3+ keyword/statutory matches.

**Limitations**:
- No expert legal validation of relevance judgments
- Keyword overlap does not equal legal relevance
- A case may be legally relevant but use different terminology

### How to Upgrade to Human Labels

1. Open `ground_truth.json`
2. For each query, review the `relevant_cases` list against the indexed HistoricalCase records
3. Move correctly labeled cases to `highly_relevant_cases`
4. Remove incorrectly labeled cases
5. Change `"label_confidence": "proxy"` to `"label_confidence": "human"`
6. Re-run `python evaluation/run_evaluation.py`

## Baselines

| ID | Name | Description |
|---|---|---|
| B1 | `bm25_only` | BM25Plus lexical scoring only |
| B2 | `dense_only` | Cosine similarity on embeddings only |
| B3 | `hybrid` | 0.70 * dense + 0.30 * BM25 |
| B4 | `full_reranker` | Full six-signal reranker |

All baselines return ranked historical **case IDs** (not chunk IDs) with per-case diversification.

## Ablation Configurations

| Config | Signals Active |
|---|---|
| A0 | Dense only |
| A1 | Dense + BM25 |
| A2 | Dense + BM25 + Statutory |
| A3 | Dense + BM25 + Statutory + Issue |
| A4 | Dense + BM25 + Statutory + Issue + Citation |
| A5 | All 6 signals (full system) |

## What Is NOT Evaluated (Honest Gaps)

| Gap | Reason | What's Needed |
|---|---|---|
| Certified IR metrics | No human-annotated labels | Expert annotation of 50+ queries |
| Real LLM provenance | Uses MockLLMProvider | Claude API access + live analysis |
| Semantic embedding quality | Uses HashingVectorizer fallback | Verify SentenceTransformers is active |
| Cross-jurisdiction robustness | Only SC queries evaluated | Add High Court query set |

## Running Evaluation Tests

```bash
# From backend/ directory — existing tests
python -m pytest tests/ -v

# From evaluation/ directory — new edge case tests
cd evaluation
python -m pytest tests/test_edge_cases.py -v
```
