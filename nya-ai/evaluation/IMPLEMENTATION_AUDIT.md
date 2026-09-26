# Nya(AI) — Implementation Audit for Mid-Sem Evaluation Framework

Generated: September 2026  
Auditor: Evaluation Framework Build Process

---

## 1. What Already Exists (Reusable Components)

### Database & Schema (`backend/app/models.py`)
| Model | Purpose | Relevant Fields |
|---|---|---|
| `HistoricalCase` | Indexed Indian court judgments | `id, case_name, court, year, citation, summary, outcome, legal_sections, laws, cites_count, cited_by_count, is_synthetic` |
| `Chunk` | Text fragment from a judgment | `id, historical_case_id, text, embedding, section_type, char_start, char_end, page_number` |
| `Document` | Source document for a case | `id, case_id, historical_case_id, filename, full_text, num_chunks` |
| `Case` | Active user workspace | `id, title, facts, issues, provisions, arguments, court, case_type` |
| `Analysis` | Stored analysis result | `id, case_id, analysis_type, content, provider` |
| `Citation` | Provenance link to source chunk | `id, analysis_id, ref_key, chunk_id, historical_case_id, case_name, excerpt, similarity_score` |

**DB location**: `backend/data/nya.db` (SQLite, ~124 MB)  
**ORM**: SQLAlchemy - `backend/app/database.py`  
**Session factory**: `SessionLocal`

---

### Retrieval Layer (`backend/app/services/retrieval.py`)
- `retrieve_historical_chunks(db, query_text, top_k, per_case_cap)` -> `List[RetrievedChunk]`
- `RetrievedChunk` dataclass: `chunk, semantic_score, keyword_score, hybrid_score`
- Internal formula: `hybrid = 0.70 * semantic + 0.30 * bm25`
- Uses `BM25Plus` from `rank_bm25`
- Uses `cosine_similarity` from embeddings module
- Applies per-case diversification cap

**Can be reused directly** for baselines B3 (hybrid) and as base for B1/B2/B4.

---

### Reranker (`backend/app/services/reranker.py`)
- `rerank_candidates(db, query_case, candidates, top_k, weights)` -> `List[RerankedCandidate]`
- Weights: semantic=0.35, lexical=0.20, statute=0.20, issue=0.10, citation=0.08, court=0.07
- `RerankedCandidate.to_dict()` returns full score breakdown + explanation

**Can be reused directly** for baseline B4 (full six-signal reranker).

---

### Embedding Provider (`backend/app/services/embeddings.py`)
- `get_embedding_provider()` -> EmbeddingProvider
- Default: `SentenceTransformerEmbeddingProvider` (all-MiniLM-L6-v2, 384d)
- Fallback: `LocalHashingEmbeddingProvider` (512d, offline deterministic)
- `cosine_similarity(a, b)` utility function

---

### Legal Structure Extractor (`backend/app/services/legal_structure.py`)
- `extract_legal_sections_structure(text)` -> dict with facts, legal_issues, petitioner_arguments, respondent_arguments, evidence, reasoning, decision, relief, laws, sections
- `extract_legal_entities(text)` -> sections, laws, citations, judges
- Rule-based regex patterns; no ML

---

### Citation Graph (`backend/app/services/citation_graph.py`)
- `get_citation_graph(db)` -> `LegalCitationGraph` singleton
- `get_authority_score(case_id)` -> float (in-degree normalized)
- Built with NetworkX DiGraph

---

### Existing Evaluation Script (`backend/scripts/evaluate.py`)
- 4 hardcoded benchmark queries with keyword-based precision
- LIMITATION: Uses keyword matching as "ground truth" proxy - not true IR metrics
- LIMITATION: No real relevance labels
- Will be superseded by new framework; existing script is preserved.

---

### Existing Tests (`backend/tests/`)
| File | Tests |
|---|---|
| `test_v2.py` | chunking, case extraction, hybrid retrieval, background workflow |
| `test_api_integration.py` | all API endpoints via TestClient |

Target: All 28 existing tests must continue passing.

---

## 2. What Must Be Added

| Component | File | Purpose |
|---|---|---|
| Evaluation dataset | `evaluation/queries.json` | 20 structured queries from real indexed cases |
| Ground truth | `evaluation/ground_truth.json` | Relevance annotations (proxy auto-labels) |
| Structure annotations | `evaluation/structure_annotations.json` | Per-case extraction ground truth (format defined) |
| IR Metrics | `evaluation/metrics.py` | P@K, Recall@K, MRR, NDCG@K, Hit Rate@K |
| Baselines | `evaluation/baselines.py` | BM25-only, Dense-only, Hybrid, Full-Reranker |
| Ablation | `evaluation/ablation.py` | 6 progressive signal configurations A0-A5 |
| Weight Sensitivity | `evaluation/weight_sensitivity.py` | 5 weight configurations |
| Structure Eval | `evaluation/structure_evaluator.py` | Precision/Recall/F1 per field |
| Provenance Validator | `evaluation/provenance_validator.py` | Citation chain verification |
| Performance | `evaluation/performance.py` | Latency benchmarks |
| Case Study | `evaluation/case_study.py` | End-to-end pipeline with intermediate outputs |
| Main runner | `evaluation/run_evaluation.py` | One-command orchestrator |
| Failure tests | `evaluation/tests/test_edge_cases.py` | Edge/failure cases |
| Figures generator | `evaluation/generate_figures.py` | Charts from result JSON/CSV |
| Final report | `evaluation/MIDSEM_EVALUATION_REPORT.md` | Academic-style evaluation report |

---

## 3. Exact Functions Used by Evaluation Framework

```
evaluation/ -> backend/app/services/retrieval.py
  retrieve_historical_chunks(db, query_text, top_k, per_case_cap)
  aggregate_by_historical_case(retrieved)

evaluation/ -> backend/app/services/reranker.py
  rerank_candidates(db, query_case, candidates, top_k, weights)

evaluation/ -> backend/app/services/embeddings.py
  get_embedding_provider()
  cosine_similarity(a, b)

evaluation/ -> backend/app/services/legal_structure.py
  extract_legal_sections_structure(text)
  extract_legal_entities(text)

evaluation/ -> backend/app/services/citation_graph.py
  get_citation_graph(db)

evaluation/ -> backend/app/services/analysis_service.py
  run_similar_cases(db, case)

evaluation/ -> backend/app/database.py
  SessionLocal, Base

evaluation/ -> backend/app/models.py
  HistoricalCase, Chunk, Document, Case, Analysis, Citation
```

---

## 4. Ground Truth Approach

**Problem**: No manual relevance annotations exist.

**Solution Adopted**:
1. Build queries from real indexed HistoricalCase records (using their summary, laws, case_type)
2. Auto-label "relevant" cases using statutory overlap + keyword co-occurrence (proxy labels)
3. Mark all auto-labels as "label_confidence": "proxy" - NOT ground truth
4. Provide annotation schema so human experts can upgrade labels
5. All metrics computed with proxy labels are clearly marked [PROXY LABELS] in reports
6. Do NOT report inflated metrics as if they were from manual annotation

---

## 5. Known Limitations

1. No human-annotated relevance labels - all retrieval metrics use keyword/statutory proxy labels
2. Embedding provider may vary - results depend on whether sentence-transformers or local hashing is active
3. Citation graph is partially synthetic - edge construction uses year+court heuristics
4. LLM synthesis uses MockLLMProvider by default - provenance test reflects mock output
5. existing evaluate.py uses keyword matching, not case-level relevance - preserved but superseded
