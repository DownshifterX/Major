from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=schemas.CaseOut)
def create_case(payload: schemas.CaseCreate, db: Session = Depends(get_db)):
    case = models.Case(
        title=payload.title,
        case_number=payload.case_number,
        court=payload.court,
        jurisdiction=payload.jurisdiction,
        judge=payload.judge,
        date_of_judgment=payload.date_of_judgment,
        date_of_filing=payload.date_of_filing,
        case_type=payload.case_type,
        legal_domain=payload.legal_domain,
        parties=[p.dict() for p in (payload.parties or [])],
        provisions=payload.provisions or [],
        facts=payload.facts or "",
        issues=payload.issues or "",
        arguments=payload.arguments or {},
        evidence=payload.evidence or [],
        procedural_history=payload.procedural_history or "",
        judgment=payload.judgment or "",
        timeline=payload.timeline or [],
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.get("", response_model=list[schemas.CaseOut])
def list_cases(db: Session = Depends(get_db)):
    return db.query(models.Case).order_by(models.Case.created_at.desc()).all()


@router.get("/{case_id}", response_model=schemas.CaseOut)
def get_case(case_id: str, db: Session = Depends(get_db)):
    case = db.query(models.Case).get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.patch("/{case_id}", response_model=schemas.CaseOut)
def update_case(case_id: str, payload: schemas.CaseUpdate, db: Session = Depends(get_db)):
    case = db.query(models.Case).get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    data = payload.dict(exclude_unset=True)
    if "parties" in data and data["parties"] is not None:
        data["parties"] = [p if isinstance(p, dict) else p.dict() for p in data["parties"]]
    for k, v in data.items():
        setattr(case, k, v)
    db.commit()
    db.refresh(case)
    return case


@router.delete("/{case_id}")
def delete_case(case_id: str, db: Session = Depends(get_db)):
    case = db.query(models.Case).get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    db.delete(case)
    db.commit()
    return {"deleted": True}


@router.get("/{case_id}/dashboard-stats")
def dashboard_stats(case_id: str, db: Session = Depends(get_db)):
    case = db.query(models.Case).get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return {
        "documents": db.query(models.Document).filter(models.Document.case_id == case_id).count(),
        "chunks": db.query(models.Chunk).filter(models.Chunk.case_id == case_id).count(),
        "analyses": db.query(models.Analysis).filter(models.Analysis.case_id == case_id).count(),
        "status": case.status,
    }
