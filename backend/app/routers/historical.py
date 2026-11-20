import math
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from .. import models
from ..database import get_db

router = APIRouter(prefix="/historical-cases", tags=["historical"])

@router.get("")
def list_historical_cases(
    page: int = 1,
    limit: Optional[int] = None,
    q: Optional[str] = None,
    envelope: bool = False,
    response: Response = None,
    db: Session = Depends(get_db),
):
    query = (
        db.query(
            models.HistoricalCase.id, models.HistoricalCase.case_name, models.HistoricalCase.court,
            models.HistoricalCase.year, models.HistoricalCase.citation, models.HistoricalCase.case_type,
            models.HistoricalCase.outcome, models.HistoricalCase.summary, models.HistoricalCase.is_synthetic,
            models.HistoricalCase.source_note, func.count(models.Document.id).label("document_count"),
        )
        .outerjoin(models.Document, models.Document.historical_case_id == models.HistoricalCase.id)
        .group_by(models.HistoricalCase.id)
    )

    if q and q.strip():
        term = f"%{q.strip()}%"
        query = query.filter(
            models.HistoricalCase.case_name.ilike(term) |
            models.HistoricalCase.court.ilike(term) |
            models.HistoricalCase.citation.ilike(term) |
            models.HistoricalCase.case_type.ilike(term)
        )

    # Fast total count
    total = query.count() if (q and q.strip()) else db.query(func.count(models.HistoricalCase.id)).scalar()

    query = query.order_by(models.HistoricalCase.case_name.asc())

    if limit is not None and limit > 0:
        page = max(1, page)
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        total_pages = math.ceil(total / limit) if total > 0 else 1
    else:
        total_pages = 1
        page = 1
        limit = total

    rows = query.all()
    items = [{
        "id": r[0], "case_name": r[1], "court": r[2], "year": r[3],
        "citation": r[4], "case_type": r[5], "outcome": r[6], "summary": r[7],
        "is_synthetic": r[8], "source_note": r[9], "document_count": r[10],
    } for r in rows]

    if response:
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Page"] = str(page)
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Total-Pages"] = str(total_pages)

    if envelope:
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": total_pages,
        }
    return items

@router.get("/{historical_case_id}")
def get_historical_case(historical_case_id: str, db: Session = Depends(get_db)):
    hc = db.get(models.HistoricalCase, historical_case_id)
    if not hc: raise HTTPException(404, "Historical case not found")
    return {
        "id": hc.id, "case_name": hc.case_name, "court": hc.court, "year": hc.year,
        "citation": hc.citation, "case_type": hc.case_type, "outcome": hc.outcome,
        "summary": hc.summary, "is_synthetic": hc.is_synthetic, "source_note": hc.source_note,
        "documents": [{"id": d.id, "filename": d.filename, "doc_type": d.doc_type, "full_text": d.full_text} for d in hc.documents],
    }
