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
        issues=payload.issues or payload.charges_issues or "",
        arguments=payload.arguments or {},
        evidence=payload.evidence or [],
        procedural_history=payload.procedural_history or "",
        judgment=payload.judgment or "",
        timeline=payload.timeline or [],
    )
    if payload.arguments_text and not payload.arguments:
        case.arguments_text = payload.arguments_text
    if payload.evidence_text and not payload.evidence:
        case.evidence_text = payload.evidence_text

    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.post("/from-url", response_model=schemas.CaseOut)
def import_case_from_url(payload: schemas.UrlImportRequest, db: Session = Depends(get_db)):
    """Acquires a case judgment from URL (e.g. Indian Kanoon) and automatically extracts its legal profile."""
    from ..services.document_acquisition import acquire_document_from_url
    from ..services.legal_structure import extract_legal_sections_structure
    from ..services.indexing import index_document

    result = acquire_document_from_url(payload.url)
    if result["status"] == "failed":
        raise HTTPException(400, f"Failed to acquire document from URL: {result.get('error')}")

    clean_text = result["clean_text"]
    title = result["title"] or "Imported Case"
    structured = extract_legal_sections_structure(clean_text)

    case = models.Case(
        title=title,
        court="Supreme Court of India" if "supreme court" in clean_text.lower() else "High Court",
        case_type=payload.case_type or "General",
        facts=structured["facts"],
        issues="\n".join(structured["legal_issues"]) if isinstance(structured["legal_issues"], list) else str(structured["legal_issues"]),
        arguments={
            "petitioner": structured.get("petitioner_arguments", []),
            "respondent": structured.get("respondent_arguments", []),
        },
        evidence=[{"description": str(e)} for e in structured.get("evidence", [])],
        provisions=[{"act": l, "section": ""} for l in structured.get("laws", [])] + [{"act": "", "section": s} for s in structured.get("sections", [])],
        judgment=structured.get("decision", ""),
        procedural_history=f"Acquired from {payload.url}",
        status="indexed",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    # Save document record
    doc = models.Document(
        case_id=case.id,
        filename=f"url_import_{case.id}.txt",
        doc_type="judgment",
        source="url_acquisition",
        source_url=payload.url,
        document_status="ready",
        document_hash=result["document_hash"],
        full_text=clean_text,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    index_document(db, doc)
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


@router.get("/{case_id}/citation-graph")
def case_citation_graph(case_id: str, db: Session = Depends(get_db)):
    """Returns the precedent citation graph for the case."""
    from ..services.citation_graph import get_citation_graph
    graph = get_citation_graph(db)
    
    # If this is a live user case, get graph for its top similar historical case
    top_hist = db.query(models.HistoricalCase).first()
    target_id = top_hist.id if top_hist else case_id
    subgraph = graph.get_case_citation_subgraph(target_id, depth=2)
    return subgraph
