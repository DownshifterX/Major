"""Seed the historical case corpus from backend/data/historical_cases/*.txt

Each file has a small header block of KEY: value lines, a '---' separator,
then the judgment body text. Run with:

    python -m scripts.seed_historical_cases
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal, Base, engine
from app import models
from app.services.indexing import index_document

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "historical_cases"


def parse_case_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    header_part, _, body = text.partition("\n---\n")
    header = {}
    for line in header_part.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
    return {
        "case_name": header.get("CASE_NAME", path.stem),
        "court": header.get("COURT"),
        "year": int(header["YEAR"]) if header.get("YEAR", "").isdigit() else None,
        "citation": header.get("CITATION"),
        "case_type": header.get("CASE_TYPE"),
        "outcome": header.get("OUTCOME", ""),
        "body": body.strip(),
    }


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = db.query(models.HistoricalCase).count()
        if existing > 0:
            print(f"Historical corpus already has {existing} cases. Skipping seed. "
                  f"Delete data/nya.db to reseed from scratch.")
            return

        if not CORPUS_DIR.exists():
            print(f"No corpus directory found at {CORPUS_DIR}")
            return

        count = 0
        for path in sorted(CORPUS_DIR.glob("*.txt")):
            parsed = parse_case_file(path)
            hc = models.HistoricalCase(
                case_name=parsed["case_name"],
                court=parsed["court"],
                year=parsed["year"],
                citation=parsed["citation"],
                case_type=parsed["case_type"],
                outcome=parsed["outcome"],
                summary=parsed["body"][:600],
                is_synthetic=True,
                source_note="Synthetic demo corpus generated for the Nya(AI) PoC. Not a real judgment — replace with a real, licensed case corpus before any production use.",
            )
            db.add(hc)
            db.commit()
            db.refresh(hc)

            document = models.Document(
                historical_case_id=hc.id,
                filename=path.name,
                doc_type="judgment",
                source="seed_corpus",
                full_text=parsed["body"],
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            n_chunks = index_document(db, document)
            print(f"Seeded '{hc.case_name}' ({n_chunks} chunks)")
            count += 1

        print(f"Done. Seeded {count} historical cases.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
