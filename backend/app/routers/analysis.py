from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services import analysis_service as svc

router = APIRouter(prefix="/cases", tags=["analysis"])


def _get_case(db: Session, case_id: str) -> models.Case:
    case = db.get(models.Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.post("/{case_id}/analyze")
def analyze_case(case_id: str, db: Session = Depends(get_db)):
    """Runs the full P0/P1 analysis pipeline in one call."""
    case = _get_case(db, case_id)
    results = {
        "similar_cases": svc.run_similar_cases(db, case),
        "legal_provisions": svc.run_legal_provisions(db, case),
        "prosecution_defence": svc.run_prosecution_defence(db, case),
        "evidence_gaps": svc.run_evidence_gaps(db, case),
        "contradictions": svc.run_contradictions(db, case),
    }
    case.status = "analyzed"
    db.commit()
    return results


@router.get("/{case_id}/similar-cases")
def get_similar_cases(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_similar_cases(db, case)


@router.get("/{case_id}/legal-provisions")
def get_legal_provisions(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_legal_provisions(db, case)


@router.get("/{case_id}/prosecution-defence")
def get_prosecution_defence(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_prosecution_defence(db, case)


@router.get("/{case_id}/evidence-gaps")
def get_evidence_gaps(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_evidence_gaps(db, case)


@router.get("/{case_id}/contradictions")
def get_contradictions(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_contradictions(db, case)


@router.post("/{case_id}/compare")
def compare_case(case_id: str, payload: schemas.CompareRequest, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    try:
        return svc.run_comparison(db, case, payload.historical_case_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/{case_id}/brief")
def get_brief(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    return svc.run_brief(db, case)


from fastapi.responses import Response
from ..services import export as export_service

@router.get("/{case_id}/brief/export")
def export_brief(case_id: str, db: Session = Depends(get_db)):
    case = _get_case(db, case_id)
    pdf_bytes = export_service.generate_brief_pdf(db, case)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=brief_{case_id}.pdf"}
    )


@router.get("/{case_id}/analysis")
def list_analyses(case_id: str, db: Session = Depends(get_db)):
    _get_case(db, case_id)
    rows = (db.query(models.Analysis)
            .filter(models.Analysis.case_id == case_id)
            .order_by(models.Analysis.created_at.desc()).all())
    return [{"id": r.id, "analysis_type": r.analysis_type, "provider": r.provider,
             "created_at": r.created_at.isoformat(), "content": r.content} for r in rows]
