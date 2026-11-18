# System Parameters & Working Spec

Single source of truth for all pipeline constants, hyperparameters, and runtime configs.

---

### 1. Ingestion & Chunking
* **Chunk size**: `900` chars (`CHUNK_SIZE` in `backend/app/services/chunking.py`)
* **Chunk overlap**: `150` chars (`CHUNK_OVERLAP` in `backend/app/services/chunking.py`)
* **Chunk boundary**: Paragraphs (`\n\s*\n`), preserved sentences
* **Max file size**: `15 MB` (`MAX_FILE_BYTES` in `backend/app/routers/documents.py`)
* **Supported file types**: `.pdf`, `.txt`, `.md`, `.docx` (`backend/app/services/text_extraction.py`)

---

### 2. Random Seeds & Reproducibility
* **Reranker Weight Optimizer**: `1950` (`np.random.seed(1950)` in `evaluation/weight_optimizer.py` — commemorates Article 141 & Indian Constitutional precedent framework)
* **Initial Database Seed Corpus**: `6` baseline landmark cases (`backend/data/historical_cases/case_01.txt` to `case_06.txt` seeded via `backend/scripts/seed_historical_cases.py`)

---

### 3. Embeddings & Semantic Search
* **Default model**: `sentence-transformers/all-MiniLM-L6-v2` (`EMBEDDING_MODEL`)
* **Vector dimension**: `384` (BERT) | `512` fallback (`LocalHashingEmbeddingProvider`)
* **Metric**: Vectorized Cosine Similarity (`backend/app/services/retrieval.py`)
* **Lexical retrieval**: `BM25Plus` (`rank_bm25`) on case text tokens

---

### 3. Retrieval Parameters
* **Initial top_k**: `12` (configurable via `TOP_K` env, defaults to 30 for reranker stage)
* **Per-case cap**: `2` chunks per case (prevents single-case dominance)
* **Hybrid blend**: `0.65 * semantic + 0.35 * lexical`

---

### 4. 6-Signal Precedent Reranker (`config/reranker_weights.json`)
* **Active mode**: `fixed`
* **Weights**:
  * Semantic similarity: `0.35`
  * Lexical overlap: `0.20`
  * Statutory provision match: `0.20`
  * Legal issue match: `0.10`
  * Citation graph authority: `0.08`
  * Court hierarchy: `0.07`
* **Court hierarchy scores**:
  * Supreme Court: `1.0` (Art. 141 binding)
  * High Court: `0.75`
  * District Court: `0.50`
  * Tribunal: `0.40`
* **Hard constraint**: Explicitly overruled precedents are filtered out (`filter_overruled: true`)

---

### 5. Runtime Environment (`.env`)
* **Host / Port**: `127.0.0.1:8000`
* **Database**: SQLite (`backend/data/nya.db`)
* **Corpus cache**: Pickled vector matrix (`backend/data/corpus_cache.pkl`)
* **LLM Provider**: `mock` (dev/testing) | `gemini` / `groq` (production)
