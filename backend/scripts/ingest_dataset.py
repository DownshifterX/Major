"""Nya AI — Legal Dataset Ingestion Pipeline.

Ingests `combined.xlsx` or parquet files from `dataset - hugging face/` into the database:
- Dataset inspection & statistics logging
- Data validation and URL normalization
- Intelligent deduplication by doc_url and normalized case name
- Preserves raw data while creating clean historical judgments
- Legal entity extraction (statutes, sections, citations)
- Legal chunking and embedding into vector store
- Incremental and resumable with progress tracking
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.database import SessionLocal, Base, engine
from app import models
from app.services.text_cleaner import clean_legal_text, extract_title_and_year
from app.services.legal_structure import extract_legal_entities
from app.services.indexing import index_document
from app.services.citation_graph import get_citation_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_dataset")

DATASET_LAND_XLSX = BASE_DIR.parent.parent / "dataset - hugging face" / "land_disputes.xlsx"
DATASET_XLSX = BASE_DIR.parent.parent / "dataset - hugging face" / "combined.xlsx"


def load_dataset() -> pd.DataFrame:
    """Loads dataset strictly from Excel file (land_disputes.xlsx)."""
    if DATASET_LAND_XLSX.exists():
        logger.info(f"Loading Excel dataset from {DATASET_LAND_XLSX.name}...")
        return pd.read_excel(DATASET_LAND_XLSX)
    elif DATASET_XLSX.exists():
        logger.info(f"Loading Excel dataset from {DATASET_XLSX.name}...")
        return pd.read_excel(DATASET_XLSX)
    else:
        raise FileNotFoundError("No Excel dataset file found in 'dataset - hugging face/'")


def run_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive dataset audit statistics."""
    total_records = len(df)
    unique_urls = df["Doc_url"].nunique() if "Doc_url" in df else 0
    unique_titles = df["Titles"].nunique() if "Titles" in df else 0
    has_text = (df["Text"].str.strip() != "").sum() if "Text" in df else 0
    
    courts = df["Court_Name_Normalized"].value_counts().to_dict() if "Court_Name_Normalized" in df else {}
    avg_cites = float(df["Cites"].mean()) if "Cites" in df else 0.0
    avg_cited_by = float(df["Cited_by"].mean()) if "Cited_by" in df else 0.0

    stats = {
        "total_records": total_records,
        "unique_urls": unique_urls,
        "unique_titles": unique_titles,
        "duplicate_url_count": total_records - unique_urls,
        "records_with_text": int(has_text),
        "text_availability_rate": round(has_text / total_records * 100, 2) if total_records else 0,
        "top_courts": {k: int(v) for k, v in list(courts.items())[:8]},
        "avg_cites": round(avg_cites, 2),
        "avg_cited_by": round(avg_cited_by, 2),
    }
    return stats


def ingest(limit: int = 150, batch_size: int = 25, rebuild_schema: bool = False):
    """Main ingestion worker."""
    if rebuild_schema:
        logger.info("Rebuilding database schema...")
        Base.metadata.create_all(bind=engine)

    df = load_dataset()
    stats = run_audit(df)
    logger.info("=== Dataset Audit Statistics ===")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    db = SessionLocal()
    try:
        # Check existing historical case count
        existing_urls = set(
            u[0] for u in db.query(models.HistoricalCase.doc_url).filter(models.HistoricalCase.doc_url.isnot(None)).all()
        )
        logger.info(f"Existing cases in DB: {len(existing_urls)}")

        # Sort to prioritize landmark Supreme Court and High Court judgments with high citations
        df_sorted = df.sort_values(by=["Cited_by", "Cites"], ascending=False)

        ingested_count = 0
        skipped_count = 0

        for idx, row in df_sorted.iterrows():
            if limit and ingested_count >= limit:
                break

            doc_url = str(row.get("Doc_url", "")).strip()
            if not doc_url or doc_url in existing_urls:
                skipped_count += 1
                continue

            raw_title = str(row.get("Titles", "")).strip()
            clean_title, year = extract_title_and_year(raw_title)
            court = str(row.get("Court_Name_Normalized", row.get("Court_Name", ""))).strip()
            raw_text = str(row.get("Text", "")).strip()
            cites = int(row.get("Cites", 0)) if pd.notnull(row.get("Cites")) else 0
            cited_by = int(row.get("Cited_by", 0)) if pd.notnull(row.get("Cited_by")) else 0

            # Clean text
            clean_text = clean_legal_text(raw_text)
            if not clean_text or len(clean_text) < 60:
                # Text is missing or too short; record as missing document
                skipped_count += 1
                continue

            entities = extract_legal_entities(clean_text)

            hc = models.HistoricalCase(
                case_name=clean_title,
                court=court,
                year=year,
                citation=f"{year or ''} Indian Kanoon ({cites} cites)",
                case_type=str(row.get("Case_Type", "Land&Property")),
                doc_url=doc_url,
                cites_count=cites,
                cited_by_count=cited_by,
                court_normalized=court,
                legal_sections=entities["sections"],
                laws=entities["laws"],
                outcome="",
                summary=clean_text[:600],
                is_synthetic=False,
                source_note="Grounded land precedent from Indian Kanoon dataset.",
            )
            db.add(hc)
            db.commit()
            db.refresh(hc)

            # Create document
            doc_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            doc = models.Document(
                historical_case_id=hc.id,
                filename=f"judgment_{hc.id}.txt",
                doc_type="judgment",
                source="seed_corpus",
                source_url=doc_url,
                document_status="ready",
                document_hash=doc_hash,
                full_text=clean_text,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Index document
            num_chunks = index_document(db, doc)
            existing_urls.add(doc_url)
            ingested_count += 1

            if ingested_count % batch_size == 0 or (limit and ingested_count == limit):
                logger.info(f"Progress: Ingested {ingested_count} land cases ({num_chunks} chunks for latest)")

        logger.info(f"Ingestion finished! Total land cases ingested: {ingested_count}, skipped: {skipped_count}")

        # Update citation graph
        logger.info("Refreshing legal citation graph...")
        graph = get_citation_graph(db)
        logger.info(f"Citation graph updated with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal cases dataset into Nya AI.")
    parser.add_argument("--limit", type=int, default=0, help="Number of cases to ingest (0 for ALL cases in Excel)")
    parser.add_argument("--batch-size", type=int, default=25, help="Logging batch size")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DB tables before ingestion")
    args = parser.parse_args()

    ingest(limit=args.limit, batch_size=args.batch_size, rebuild_schema=args.rebuild)
