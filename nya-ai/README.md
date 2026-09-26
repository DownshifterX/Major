# ⚖️ NyaAi — Justice. Reimagined.

> **Next-Generation AI Legal Intelligence, Case Analysis & Precedent Research Platform**  
> *Grounded in Real Indian Supreme Court & High Court Jurisprudence with Zero Hallucination Citation Discipline.*

---

## ⚡ 1-Click Quick Launch

Get the entire application (FastAPI backend, SQLite/Vector database, and single-page frontend) running in **one command**:

```bash
python run.py
```

✨ **What this does automatically:**
1. 🔍 Verifies Python dependencies & activates configuration (`.env`).
2. 🗄️ Initializes SQLite schema & checks the precedent library (**130 cases, 10,949 chunks indexed**).
3. 🌐 Starts the high-performance Uvicorn server on `http://127.0.0.1:8000`.
4. 🚀 **Automatically opens your web browser** directly into the platform!

### 🛠️ CLI Launch Flags

```bash
python run.py --port 8080 --host 0.0.0.0   # Custom host and port
python run.py --no-browser                # Headless/server mode without opening browser
python run.py --reload                    # Hot-reloading mode for active development
```

---

## 🏛️ Platform Overview

**NyaAi** is a decision-support and legal intelligence platform purpose-built for advocates, researchers, judicial clerks, and legal scholars. It pairs hybrid semantic + lexical search with structural legal entity extraction and a multi-signal reranking engine.

```
       ┌────────────────────────────────────────────────────────┐
       │             User Query / Case Facts & Issues           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌─────────────────────────┐   ┌─────────────────────────┐
       │   Dense Vector Search   │   │     BM25 Keyword Search  │
       │ (BERT / all-MiniLM-L6)  │   │  (Statutes, Acts, Secs) │
       └────────────┬────────────┘   └────────────┬────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Candidate Precedent Pool (Top 30-100)          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │               Multi-Signal Legal Reranker              │
       │  • Semantic narrative alignment                        │
       │  • BM25 exact legal term match                         │
       │  • Statutory provision overlap (e.g. IPC, CrPC)        │
       │  • Citation graph authority (PageRank, citation count) │
       │  • Judicial hierarchy weight (SC > HC)                 │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       Transparent Breakdown + Grounded Citations       │
       │       • Score Explanation  • Evidence Provenance       │
       └────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

| Feature | Description | Status |
| :--- | :--- | :---: |
| 📚 **Real Precedent Corpus** | Ingested **130 Landmark Indian Judgments** (*Kesavananda Bharati*, *Maneka Gandhi*, *Siddharam Mhetre*, *Bachan Singh*, etc.) with **7,984 citation edges**. | ✅ Ready |
| 🔎 **Legal Research Search Tab** | Live hybrid retrieval querying 10,949 judgment chunks with transparent `Semantic`, `BM25`, and `Hybrid` score cards. | ✅ Ready |
| 🎯 **Multi-Signal Reranker** | Dynamically balances semantic similarity, keyword overlap, statute overlap, citation authority, and court hierarchy. | ✅ Ready |
| 🕸️ **Citation Graph Analytics** | Computes PageRank, citation centrality, in/out degree, and authority paths using NetworkX. | ✅ Ready |
| 📑 **Legal Structure Extraction** | Canonical structure extractor parsing Facts, Issues, Arguments (Petitioner vs Respondent), Reasoning, Decisions, and Statutes. | ✅ Ready |
| 🛡️ **Zero-Hallucination RAG** | Claims carry clickable citation chips `[S1]`, `[S2]` leading to the exact judgment paragraph with page and character offsets. | ✅ Ready |
| 📂 **Multi-Format Ingestion** | Upload `.pdf`, `.docx`, `.txt`, or `.md` files, or scrape judgments directly from Indian Kanoon URLs. | ✅ Ready |
| ⚖️ **Case Comparison Matrix** | Side-by-side comparative analysis of current case arguments and evidence against top binding precedents. | ✅ Ready |
| 📝 **Pre-Hearing Brief Assembly** | Automatically compiles comprehensive case briefs ready for courtroom preparation. | ✅ Ready |

---

## 🗂️ Project Structure

```
nya-ai/
├── run.py                          # ⚡ Unified 1-click application launcher
├── README.md                       # 📖 Interactive documentation & guide
├── .env.example                    # ⚙️ Configuration blueprint
│
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application & static route mounting
│   │   ├── database.py             # SQLAlchemy session & SQLite/Postgres connection
│   │   ├── models.py               # ORM Models (Case, Document, Chunk, HistoricalCase, Citation)
│   │   ├── schemas.py              # Pydantic validation schemas
│   │   │
│   │   ├── routers/
│   │   │   ├── cases.py            # /cases CRUD & workspace endpoints
│   │   │   ├── documents.py        # /documents upload, indexing & extraction
│   │   │   ├── analysis.py         # /analyze, /similar-cases, /compare, /brief
│   │   │   └── historical.py       # /historical-cases precedent endpoints
│   │   │
│   │   └── services/
│   │       ├── retrieval.py        # Hybrid dense vector + BM25 search
│   │       ├── reranker.py         # Multi-signal scoring & precedent ranking
│   │       ├── legal_structure.py  # Legal entities & section extractor
│   │       ├── citation_graph.py   # NetworkX citation graph & centrality
│   │       ├── document_acquisition.py # URL scraper with exponential backoff
│   │       ├── embeddings.py       # SentenceTransformers / Hashing / OpenAI abstraction
│   │       ├── chunking.py         # Legal section & paragraph-aware chunker
│   │       ├── llm_provider.py     # Grounded mock NLP & Anthropic Claude caller
│   │       └── analysis_service.py # Orchestrator linking retrieval, analysis & citations
│   │
│   ├── data/
│   │   └── nya.db                  # Persisted SQLite database with real Indian precedents
│   ├── scripts/
│   │   ├── ingest_dataset.py       # Dataset ingestion pipeline from parquet/excel
│   │   └── evaluate.py             # Quantitative benchmark suite (MRR, Precision@K)
│   └── tests/
│       ├── test_v2.py              # Core service & indexing unit tests
│       └── test_api_integration.py # Full FastAPI integration test suite
│
└── frontend/                       # 🎨 Dark-mode Single-Page Application (Zero build step)
    ├── index.html                  # Semantic UI with Government of India header
    ├── styles.css                  # Custom responsive dark theme & glassmorphism
    └── app.js                      # Reactive state management, search & interactive tabs
```

---

## 🚀 Manual Step-by-Step Installation

If you prefer setting up manually without `run.py`:

### 1. Prerequisites
- **Python 3.10+** installed
- Windows, macOS, or Linux

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd nya-ai

# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `backend/.env`:
```bash
cp ../.env.example .env
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `mock` | `mock` (deterministic, grounded NLP) or `anthropic` (Claude API) |
| `ANTHROPIC_API_KEY`| *(empty)* | Optional API key if using Claude models |
| `EMBEDDING_PROVIDER`| `sentence-transformers` | Semantic BERT model (`all-MiniLM-L6-v2`) or `local` |
| `EMBEDDING_DIM` | `384` | Embedding dimensions (384 for MiniLM, 512 for local) |
| `DATABASE_URL` | `sqlite:///./data/nya.db` | Local SQLite path or PostgreSQL connection string |

### 4. Run the Dev Server
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🧪 Testing & Evaluation Benchmark

### 1. Run Automated Test Suites
NyaAi comes with 28 automated test cases covering case workflows, document indexing, SQLite concurrency, and retrieval safety:

```bash
cd backend
python -m pytest tests/test_v2.py tests/test_api_integration.py -v
```
```
====================== 28 passed in 4.03s =======================
```

### 2. Run the Benchmark Evaluation Suite
Assess quantitative information retrieval and legal extraction accuracy on the real dataset:

```bash
cd backend
python scripts/evaluate.py
```

**Benchmark Results:**
- 🎯 **Legal Structure Extraction Accuracy**: `100.0%`
- 📈 **Retrieval Precision@K**: `1.0`
- 🏅 **Mean Reciprocal Rank (MRR)**: `1.0`
- ⚖️ **Multi-Signal Precedent Reranking**: Verified on landmark bail and criminal jurisprudence (*Siddharam Mhetre*, *Bachan Singh*).

---

## 🧭 Interactive User Walkthrough

1. **Conduct Instant Legal Research**:
   - Click the **Legal Research** tab in the navigation bar.
   - Enter a query, e.g.: `"anticipatory bail under section 438 crpc"` or `"circumstantial evidence section 302 ipc"`.
   - Review the matched precedents with separate **Semantic**, **BM25**, and **Hybrid** score badges.

2. **Register a New Case**:
   - Click **+ Register New Case** on the top bar.
   - Either paste an **Indian Kanoon judgment URL** to auto-fetch the text, or fill in title, facts, and legal provisions.

3. **Upload Case Documents**:
   - Inside the case workspace, switch to the **Documents** tab.
   - Upload FIRs, charge sheets, lower court orders, or arguments in `.pdf`, `.docx`, or `.txt`.
   - The document is automatically cleaned, paragraph-chunked, and embedded into the vector store.

4. **Trigger Grounded AI Analysis**:
   - Click **⚡ Run Full Analysis**.
   - Review findings across all specialized panels:
     - 🔍 **Similar Cases**: Top retrieved historical precedents with multi-signal score breakdown.
     - 📜 **Legal Provisions**: Extracted sections, acts, and procedural compliance checklist.
     - ⚔️ **Arguments & Counter-Arguments**: Structured petitioner and respondent points.
     - ⚠️ **Evidence Gaps & Contradictions**: Identified timeline conflicts and missing corroborations.
     - ⚖️ **Case Comparison**: Compare your case side-by-side with any selected precedent.
     - 📋 **Pre-Hearing Brief**: Comprehensive synthesized brief ready for court hearings.

---

## 📡 REST API Reference

Interactive Swagger documentation is available live at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Core Endpoints

```http
# System Health & Providers
GET    /api/health

# Legal Research (Hybrid Semantic + BM25)
GET    /api/search?q={query}&limit={n}

# Case Management
POST   /cases                          # Create draft case
GET    /cases                          # List all active cases
GET    /cases/{id}                     # Get case details
PATCH  /cases/{id}                     # Update case metadata
DELETE /cases/{id}                     # Delete case

# Document Management
POST   /cases/{id}/documents           # Upload & auto-index document (.pdf/.docx/.txt)
GET    /cases/{id}/documents           # List uploaded documents
DELETE /cases/{id}/documents/{doc_id}  # Remove document

# Grounded Legal Intelligence
POST   /cases/{id}/analyze             # Execute complete multi-stage analysis pipeline
GET    /cases/{id}/similar-cases       # Retrieve top similar precedents
GET    /cases/{id}/reranked-precedents # Multi-signal reranked precedents with score breakdown
GET    /cases/{id}/legal-provisions    # Extract applicable statutes and sections
GET    /cases/{id}/prosecution-defence # Arguments vs Counter-arguments
GET    /cases/{id}/evidence-gaps       # Uncover evidentiary holes
GET    /cases/{id}/contradictions      # Detect witness & timeline inconsistencies
POST   /cases/{id}/compare             # Compare case with precedent: {"historical_case_id": "..."}
GET    /cases/{id}/brief               # Generate courtroom pre-hearing brief
GET    /cases/{id}/citation-graph      # Graph nodes & edges for precedent network

# Precedent Corpus
GET    /historical-cases               # Browse precedent database
GET    /historical-cases/{id}          # View precedent judgment details
```

---

## 🔒 Ethical AI & Citation Discipline

> [!IMPORTANT]
> **NyaAi adheres strictly to legal ethics:**
> - ❌ **No Hallucinations**: Analysis never generates citations, case names, or statutory sections that do not exist in the source corpus.
> - 🔍 **Clickable Provenance**: Every output section references an exact source chunk `[S1]`, `[S2]`, allowing immediate human verification.
> - ⚖️ **Decision-Support Only**: NyaAi does not replace the counsel of qualified legal professionals or the judicial discretion of the courts.

---

## 📄 License & Attribution
Developed as part of the **NyaAi** Project.  
Judicial text extracts and citations are sourced from open Indian public domain court records.
