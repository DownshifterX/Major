# Nya AI — ML & NLP Pipeline

## 1. BERT & Sentence-Transformers Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2` (default 384 dimensions) or local hashing vectorizer.
- Persistence: Stored as serialized float vectors in the SQLite / PostgreSQL `chunks` table.
- Section-aware tagging: Each chunk is annotated with `section_type` (`facts`, `issues`, `arguments`, `evidence`, `laws`, `reasoning`, `decision`).

## 2. Legal Structure & Named Entity Extraction
- Extracts:
  - Canonical sections: Facts, Issues, Petitioner Arguments, Respondent Arguments, Evidence, Reasoning, Decision, Relief.
  - Legal entities: IPC/CrPC/CPC/Constitution sections, Acts and Statutes, Citations (AIR, SCC, SCR), Bench / Judges.
- Implementation: Deterministic rule-assisted extraction in offline mode, augmented with LLM generation (`generate_structured`) when Anthropic/OpenAI is configured.

## 3. Grounded RAG & Provenance
- Claims in analysis views are tagged with `[S1]`, `[S2]` citation references.
- Every citation maps back to a persisted `Citation` row containing document ID, page number, similarity score, and excerpt.
