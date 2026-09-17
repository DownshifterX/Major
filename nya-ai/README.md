# Nya(AI) — Justice. Reimagined.

An AI-assisted legal case analysis and decision-support **proof-of-concept**.
It is a real, working, end-to-end RAG (Retrieval-Augmented Generation)
application — not a mockup — built to demonstrate how a historical case
corpus can be used to analyze a current case, surface similar precedents,
compare cases, flag evidence gaps and contradictions, and assemble a
source-grounded pre-hearing brief.

> **Nya(AI) is a decision-support tool for legal professionals.** It never
> determines guilt, innocence, sentencing, or a final legal outcome, and it
> is designed to never fabricate a case, statute, quotation, or fact — every
> generated claim carries a citation back to a real source chunk that you
> can inspect.

---

## 1. What's actually implemented (P0 + most of P1)

- **V2 Features**: Automatic case extraction, background processing for large files, and DOCX/PDF support.
- Case creation & a case workspace (metadata, facts, issues, arguments, evidence, timeline)
- Document upload (.pdf / .txt / .md / .docx), text extraction, paragraph-aware chunking with character offsets
- A genuine embedding step (BERT via `sentence-transformers` by default, or `local`/`openai`) and a persisted vector store
- Hybrid retrieval (semantic cosine similarity + keyword overlap via BM25) with deduplication and per-case diversification
  historical corpus
- An `LLMProvider` abstraction with two real implementations:
  - `mock` (default): deterministic, rule-based, **grounded** extraction over
    the actually-retrieved text (regex statute/date extraction, keyword-based
    argument sorting, timeline-conflict heuristics) — runs fully offline,
    zero API cost, never invents anything not present in the source text
  - `anthropic`: calls the real Claude API when you provide an API key
- First-class **Citation** objects (case, document, page, excerpt, score) —
  every analysis persists its citations, and the UI lets you click a
  reference chip to open the actual source excerpt
- Similar-case retrieval, case comparison, legal-provision extraction,
  prosecution/defence analysis, evidence-gap analysis, contradiction
  detection, and pre-hearing brief assembly — all wired to the same
  retrieval + citation pipeline
- A clean single-page frontend (dashboard, case workspace with tabs,
  document upload, analysis views, source-inspection drawer)

## 2. What's intentionally out of scope for this PoC (P2)

Massive corpus ingestion, multi-agent orchestration, citation graphs,
knowledge graphs, enterprise RBAC, continuous monitoring, and complex
hearing workflows are not implemented. The architecture (provider
abstractions, structured citation objects, a relational schema with clean
foreign keys) is deliberately built so these can be layered on without a
rewrite.

---

## 3. Architecture

```
nya-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, mounts the frontend
│   │   ├── database.py             SQLAlchemy engine/session (SQLite by default)
│   │   ├── models.py               Case, Document, Chunk, HistoricalCase, Analysis, Citation
│   │   ├── schemas.py              Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── cases.py            /cases CRUD
│   │   │   ├── documents.py        upload, list, delete, /index
│   │   │   ├── analysis.py         /analyze, /similar-cases, /compare, /brief, ...
│   │   │   └── historical.py       browse the precedent corpus
│   │   └── services/
│   │       ├── text_extraction.py  pdf/txt/md -> plain text
│   │       ├── chunking.py         paragraph-aware chunking with overlap
│   │       ├── embeddings.py       EmbeddingProvider abstraction (local + stub)
│   │       ├── indexing.py         chunk -> embed -> persist
│   │       ├── retrieval.py        hybrid semantic+keyword retrieval, dedupe, diversify
│   │       ├── llm_provider.py     LLMProvider abstraction (mock + Anthropic)
│   │       ├── prompts.py          structured, source-grounded prompt templates
│   │       └── analysis_service.py orchestrates retrieval -> prompt -> LLM -> citations
│   ├── data/historical_cases/*.txt   synthetic demo precedent corpus (labeled as such)
│   └── scripts/seed_historical_cases.py
├── frontend/               index.html / app.js / styles.css (vanilla JS, no build step)
├── .env.example
└── README.md (this file)
```

**RAG pipeline:**

```
Historical .txt files → parse metadata → chunk → embed → store (SQLite)
Current case documents (PDF/TXT) → extract → chunk → embed → store (SQLite)

Current case → query representation → hybrid retrieval over historical chunks
            → context package (each chunk tagged with a ref_key, case, doc, page)
            → LLMProvider.generate_structured(prompt, context)
            → structured JSON result + persisted Citation rows
            → frontend renders result with clickable source chips
```

### Why this stack

- **FastAPI + SQLite + vanilla JS frontend** instead of Next.js/Postgres: for
  a same-day PoC this removes the Node build step and a Postgres server
  while keeping the code paths (SQLAlchemy models, a real embedding+vector
  step, a provider abstraction) identical to what you'd want in production.
  Swapping SQLite → Postgres is a one-line `DATABASE_URL` change since the
  schema uses standard SQLAlchemy types.
- **Local hashing embeddings** instead of a hosted embedding API: keeps the
  whole demo runnable with **zero external dependencies and zero cost**. See
  the honest limitations section below for exactly what this trades off, and
  `backend/app/services/embeddings.py` for where to plug in a real one.

---

## 4. Honest limitations (read this)

- **Embeddings are semantic.** The default `sentence-transformers` provider uses BERT models. Swap in a real embedding API (OpenAI, Voyage, Cohere, etc.) via `EmbeddingProvider` for production-quality semantic retrieval if needed, or use the original `local` hashing.
- **The default LLM provider (`mock`) is rule-based, not a language model.**
  It only ever surfaces information that is literally present in the
  retrieved sources (regex-matched statute references, dates, keyword
  overlaps). This is intentional for a zero-cost, zero-key demo, but it is
  *not* as capable as a real model at synthesis or nuanced legal reasoning.
  Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` to use real
  Claude generation through the same prompts and citation pipeline.
- **The historical corpus is 6 synthetic demo cases**, clearly labeled
  `is_synthetic: true` everywhere they appear in the API and UI. They are
  not real judgments. The system is built to scale to a real, licensed
  corpus (just add more `.txt`/`.pdf` files and re-run the seed script, or
  extend the ingestion path to your real source).
- **No auth / multi-user permissioning** — out of scope for this PoC per the
  brief; the API has no user accounts.
- Contradiction and evidence-gap detection are heuristic pattern checks
  (date-format regex, keyword presence) — sufficient to demonstrate the
  *shape* of the feature and its citation discipline, not a substitute for
  a lawyer's read of the file.

---

## 5. Running it

### Requirements
Python 3.10+.

### Setup

```bash
cd nya-ai/backend
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp ../.env.example .env     # edit if you want to enable the Anthropic provider
python -m scripts.seed_historical_cases   # loads the 6 synthetic precedent cases
```

### Run

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — this single process serves both the API
and the frontend (no separate frontend build/server needed).

Interactive API docs: **http://localhost:8000/docs**

### Environment variables (see `.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` (offline, grounded, rule-based) or `anthropic` (real Claude calls) |
| `ANTHROPIC_API_KEY` | _(empty)_ | required if `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | model string used for the Anthropic provider |
| `EMBEDDING_PROVIDER` | `local` | `local` (hashing, offline) — see limitations above |
| `EMBEDDING_DIM` | `512` | dimensionality of the local embedding |
| `DATABASE_URL` | `sqlite:///./data/nya.db` | swap for a Postgres URL to scale |
| `CORS_ORIGINS` | `*` (dev) | comma-separated allowed origins |

### Try the full workflow

1. Open the app → **New case** → fill in title/facts/issues/arguments/evidence.
2. Go to the **Documents** tab → upload a `.txt` or `.pdf` (e.g. an FIR/complaint).
3. Click **Analyze case** in the top bar.
4. Walk through **Similar Cases → Legal Analysis → Arguments → Evidence →
   Contradictions**, clicking the small reference chips to open the source
   drawer and see exactly which historical case / document / excerpt backed
   each claim.
5. **Case Comparison** tab → pick one of the retrieved historical cases →
   Compare.
6. **Pre-Hearing Brief** tab → generates a structured brief from everything
   already analyzed.

---

## 6. API reference (implemented)

```
POST   /cases                          create a case
GET    /cases                          list cases
GET    /cases/{id}                     get a case
PATCH  /cases/{id}                     update a case
DELETE /cases/{id}
GET    /cases/{id}/dashboard-stats

POST   /cases/{id}/documents           upload + auto-index a document
GET    /cases/{id}/documents
DELETE /cases/{id}/documents/{doc_id}
POST   /cases/{id}/index               re-chunk/re-embed all of a case's documents

POST   /cases/{id}/analyze             run the full P0/P1 pipeline in one call
GET    /cases/{id}/similar-cases
GET    /cases/{id}/legal-provisions
GET    /cases/{id}/prosecution-defence
GET    /cases/{id}/evidence-gaps
GET    /cases/{id}/contradictions
POST   /cases/{id}/compare             body: {"historical_case_id": "..."}
GET    /cases/{id}/brief
GET    /cases/{id}/analysis            list all stored analysis runs

GET    /historical-cases               browse the precedent corpus
GET    /historical-cases/{id}
```

---

## 7. What remains for the next phase

- Swap `LocalHashingEmbeddingProvider` for a real semantic embedding API and
  add a proper reranking step (e.g. cross-encoder) on top of the hybrid
  retrieval already in place.
- Replace the synthetic 6-case corpus with a real, licensed precedent
  database, and add a proper bulk-ingestion pipeline (the single-document
  `index_document()` path already generalizes to this).
- Move from SQLite to Postgres + a real vector extension (e.g. pgvector) as
  corpus size grows — the schema and `DATABASE_URL` config already support this.
- PDF/DOCX export of the pre-hearing brief (the data is already structured
  cleanly for this — see `run_brief()`).
- Basic auth / per-user case isolation before any multi-user deployment.
- Reranking + citation-graph features (P2 in the original brief).
