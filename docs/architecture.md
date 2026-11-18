# Nya AI — System Architecture

Nya AI is an AI-powered, evidence-backed legal research and case analysis platform designed for legal professionals reviewing Indian law and jurisprudence.

```
USER CASE (Upload PDF/TXT/DOCX or URL or Manual Input)
                  │
                  ▼
         DOCUMENT ACQUISITION
   (Cache, Hash, Indian Kanoon Scraper)
                  │
                  ▼
         TEXT CLEANING & PREPROCESSING
  (Strip Boilerplate, Clean Whitespace, Preserve Legal Sections)
                  │
                  ▼
       LEGAL STRUCTURE EXTRACTION
(Facts, Issues, Pet/Resp Arguments, Evidence, Laws, Reasoning, Decision)
                  │
                  ▼
       LEGAL-AWARE CHUNKING
 (Hierarchical windows tagged with section_type & offsets)
                  │
                  ▼
        VECTOR & BM25 INDEXING
(Sentence-Transformers BERT embeddings + BM25Plus index)
                  │
                  ▼
          HYBRID RETRIEVAL
   (Semantic Cosine + Lexical Overlap)
                  │
                  ▼
       MULTI-SIGNAL RERANKING
 (Semantic + BM25 + Statute Overlap + Issue Alignment + Citation Boost)
                  │
                  ▼
       CITATION GRAPH ANALYSIS
   (NetworkX Directed Precedent Authority Graph)
                  │
                  ▼
        CASE COMPARISON & RAG
 (Structured Grounded Prompts, Clickable Ref Chips [S1])
                  │
                  ▼
          GOV-PORTAL FRONTEND
   (Dashboard, Active Cases, Citation Explorer, Brief Export)
```

## Core Subsystems

1. **Document Acquisition**: Fetches public judgments from Indian Kanoon or case URLs with SHA-256 caching and retry logic.
2. **Text Cleaner**: Cleans and strips Indian Kanoon boilerplate and watermarks without discarding statutory references or paragraph structure.
3. **Legal Structure Extractor**: Extracts facts, issues, petitioner/respondent arguments, evidence items, laws, sections, reasoning, and decision.
4. **Retrieval Engine**: BM25Plus lexical search + SentenceTransformer semantic search.
5. **Reranker**: Transparent multi-signal reranking weighing semantic relevance, keyword overlap, statute overlap, issue similarity, court hierarchy, and citation frequency.
6. **Citation Graph**: Directed precedent graph tracking citations, cited-by counts, and precedent paths.
