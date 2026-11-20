"""
train_and_ingest_land_cases.py
=================================
Processes downloaded land case HTML/PDF files in `downloaded_cases/`, 
extracts full judgment text & metadata, indexes chunk embeddings into 
the Nya AI database, and updates legal entity graphs.

Usage:
------
    python train_and_ingest_land_cases.py                  # Ingest all downloaded cases
    python train_and_ingest_land_cases.py --limit 50       # Process first 50 downloaded files
    python train_and_ingest_land_cases.py --rebuild        # Rebuild DB schema before ingesting
"""

import os
import sys
import glob
import re
import argparse
import logging
from pathlib import Path
from bs4 import BeautifulSoup

# Ensure backend path is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from app.database import SessionLocal, Base, engine
from app import models
from app.services.text_cleaner import clean_legal_text, extract_title_and_year
from app.services.legal_structure import extract_legal_entities
from app.services.indexing import index_document
from app.services.citation_graph import get_citation_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_and_ingest_land_cases")

DOWNLOADED_DIR = SCRIPT_DIR / "downloaded_cases"


def extract_html_text(file_path: Path) -> dict:
    """Extract metadata and clean text from an IndianKanoon HTML case file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title
    title_tag = soup.find("title") or soup.find("h1")
    raw_title = title_tag.get_text().strip() if title_tag else file_path.stem

    # Extract court name if available
    court_div = soup.find("div", class_="doc_court") or soup.find("div", class_="court")
    court_name = court_div.get_text().strip() if court_div else "High Court / Supreme Court"

    # Extract judgment body text
    judg_div = soup.find("div", class_="judgments") or soup.find("div", id="judg_text")
    if judg_div:
        text = judg_div.get_text(separator="\n")
    else:
        # Fallback to body
        text = soup.body.get_text(separator="\n") if soup.body else soup.get_text(separator="\n")

    return {
        "title": raw_title,
        "court": court_name,
        "text": text
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest downloaded land case files into Nya AI knowledge base.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild database tables before ingesting")
    args = parser.parse_args()

    if args.rebuild:
        logger.info("Rebuilding database schema...")
        Base.metadata.create_all(bind=engine)

    if not DOWNLOADED_DIR.exists():
        logger.error(f"Downloaded cases directory not found at {DOWNLOADED_DIR}")
        sys.exit(1)

    html_files = sorted(list(DOWNLOADED_DIR.glob("*.html"))) + sorted(list(DOWNLOADED_DIR.glob("*.htm")))
    logger.info(f"Found {len(html_files)} downloaded HTML case files in {DOWNLOADED_DIR.name}")

    if not html_files:
        logger.warning("No HTML files found to process.")
        sys.exit(0)

    db = SessionLocal()
    try:
        # Fetch existing URLs/filenames to avoid duplicate ingestion
        existing_filenames = set(
            f[0] for f in db.query(models.Document.filename).all()
        )

        processed_count = 0
        skipped_count = 0

        for file_path in html_files:
            if args.limit and processed_count >= args.limit:
                break

            filename = file_path.name
            if filename in existing_filenames:
                skipped_count += 1
                continue

            extracted = extract_html_text(file_path)
            clean_title, year = extract_title_and_year(extracted["title"])
            clean_text = clean_legal_text(extracted["text"])

            if not clean_text or len(clean_text) < 100:
                logger.warning(f"Skipping {filename}: Insufficient text content extracted.")
                skipped_count += 1
                continue

            # Extract legal entities (sections, statutes, laws)
            entities = extract_legal_entities(clean_text)

            # Create HistoricalCase record
            hc = models.HistoricalCase(
                case_name=clean_title,
                court=extracted["court"],
                year=year,
                citation=f"{year or ''} Downloaded Land Case",
                case_type="Land & Property Dispute",
                doc_url=str(file_path),
                cites_count=1,
                cited_by_count=0,
                court_normalized=extracted["court"],
                legal_sections=entities["sections"],
                laws=entities["laws"],
                outcome="Judgment delivered; land rights determined.",
                summary=clean_text[:600],
                is_synthetic=False,
                source_note="Ground truth downloaded land case judgment.",
            )
            db.add(hc)
            db.commit()
            db.refresh(hc)

            # Create Document record
            doc = models.Document(
                historical_case_id=hc.id,
                filename=filename,
                doc_type="judgment",
                source="downloaded_cases",
                source_url=str(file_path),
                document_status="ready",
                full_text=clean_text,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Generate chunks and embed vectors for RAG retrieval
            num_chunks = index_document(db, doc)
            processed_count += 1
            existing_filenames.add(filename)

            logger.info(f"[{processed_count}/{len(html_files)}] Indexed '{clean_title[:50]}...' -> {num_chunks} chunks.")

        logger.info(f"Ingestion finished! Successfully indexed {processed_count} downloaded land cases. (Skipped: {skipped_count})")

        # Refresh citation graph
        logger.info("Refreshing legal citation graph...")
        graph = get_citation_graph(db)
        logger.info(f"Citation graph updated: {len(graph.graph.nodes)} nodes, {len(graph.graph.edges)} edges.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
