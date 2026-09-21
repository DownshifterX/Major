# Nya AI — Data Pipeline

## Dataset Overview
- **Source**: `dataset - hugging face/` (`part1.parquet`, `part2.parquet`, `combined.xlsx`)
- **Total Records**: 11,970 Indian court judgments.
- **Top Courts**: Supreme Court of India (1,600), Delhi High Court (799), Bombay High Court (740), Madras High Court (735), Patna High Court (727).
- **Average Citations**: 22.64 cites, 42.4 times cited by.

## Ingestion Pipeline
```
Parquet / Excel
      │
      ▼
Dataset Audit & Normalization
      │
      ▼
Intelligent Deduplication (by Doc_url and title hash)
      │
      ▼
Text Cleaning & Boilerplate Stripping (text_cleaner.py)
      │
      ▼
Entity Extraction (Sections, Acts, Citations, Judges)
      │
      ▼
Relational Storage (historical_cases table)
      │
      ▼
Section-Aware Chunking & BERT Embeddings (chunks table)
      │
      ▼
Citation Network Construction (NetworkX DiGraph)
```

## Running Ingestion
```bash
python scripts/ingest_dataset.py --limit 50 --batch-size 10
```
- `--limit`: Max number of cases to ingest (prioritizes high-citation landmark cases).
- `--batch-size`: Batch size for progress updates.
- `--rebuild`: Optional flag to drop and re-create database tables.
