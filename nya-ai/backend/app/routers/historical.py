from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/historical-cases", tags=["historical"])


from sqlalchemy import func

@router.get("")
def list_historical_cases(db: Session = Depends(get_db)):
    rows = (
        db.query(
            models.HistoricalCase.id,
            models.HistoricalCase.case_name,
            models.HistoricalCase.court,
            models.HistoricalCase.year,
            models.HistoricalCase.citation,
            models.HistoricalCase.case_type,
            models.HistoricalCase.outcome,
            models.HistoricalCase.summary,
            models.HistoricalCase.is_synthetic,
            models.HistoricalCase.source_note,
            func.count(models.Document.id).label("document_count"),
        )
        .outerjoin(models.Document, models.Document.historical_case_id == models.HistoricalCase.id)
        .group_by(models.HistoricalCase.id)
        .order_by(models.HistoricalCase.case_name.asc())
        .all()
    )
    return [{
        "id": r[0], "case_name": r[1], "court": r[2], "year": r[3],
        "citation": r[4], "case_type": r[5], "outcome": r[6],
        "summary": r[7], "is_synthetic": r[8], "source_note": r[9],
        "document_count": r[10],
    } for r in rows]


@router.get("/{historical_case_id}")
def get_historical_case(historical_case_id: str, db: Session = Depends(get_db)):
    hc = db.query(models.HistoricalCase).get(historical_case_id)
    if not hc:
        raise HTTPException(404, "Historical case not found")
    return {
        "id": hc.id, "case_name": hc.case_name, "court": hc.court, "year": hc.year,
        "citation": hc.citation, "case_type": hc.case_type, "outcome": hc.outcome,
        "summary": hc.summary, "is_synthetic": hc.is_synthetic, "source_note": hc.source_note,
        "documents": [{"id": d.id, "filename": d.filename, "doc_type": d.doc_type,
                       "full_text": d.full_text} for d in hc.documents],
    }
