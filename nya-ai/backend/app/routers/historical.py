from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/historical-cases", tags=["historical"])


@router.get("")
def list_historical_cases(db: Session = Depends(get_db)):
    rows = db.query(models.HistoricalCase).all()
    return [{
        "id": r.id, "case_name": r.case_name, "court": r.court, "year": r.year,
        "citation": r.citation, "case_type": r.case_type, "outcome": r.outcome,
        "summary": r.summary, "is_synthetic": r.is_synthetic, "source_note": r.source_note,
        "document_count": len(r.documents),
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
