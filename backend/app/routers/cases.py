from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/cases", tags=["cases"])

@router.post("", response_model=schemas.CaseOut)
def create_case(payload: schemas.CaseCreate, db: Session = Depends(get_db)):
    from ..services.legal_structure import is_land_case
    full_text = f"{payload.title} {payload.facts or ''} {payload.issues or payload.charges_issues or ''}"
    if not is_land_case(text=full_text, case_type=payload.case_type or ""):
        raise HTTPException(status_code=400, detail="NyaAi is currently designed for land cases. You can expect future updates to serve other case types.")

    case = models.Case(
        title=payload.title, case_number=payload.case_number, court=payload.court,
        jurisdiction=payload.jurisdiction, judge=payload.judge, date_of_judgment=payload.date_of_judgment,
        date_of_filing=payload.date_of_filing, case_type=payload.case_type, legal_domain=payload.legal_domain,
        parties=[p.model_dump() for p in (payload.parties or [])], provisions=payload.provisions or [],
        facts=payload.facts or "", issues=payload.issues or payload.charges_issues or "",
        arguments=payload.arguments or {}, evidence=payload.evidence or [],
        procedural_history=payload.procedural_history or "", judgment=payload.judgment or "",
        timeline=payload.timeline or [],
    )
    if payload.arguments_text and not payload.arguments: case.arguments_text = payload.arguments_text
    if payload.evidence_text and not payload.evidence: case.evidence_text = payload.evidence_text

    db.add(case); db.commit(); db.refresh(case)
    return case

@router.post("/from-url", response_model=schemas.CaseOut)
def import_case_from_url(payload: schemas.UrlImportRequest, db: Session = Depends(get_db)):
    from ..services.document_acquisition import acquire_document_from_url
    from ..services.legal_structure import extract_legal_sections_structure, is_land_case
    from ..services.indexing import index_document

    res = acquire_document_from_url(payload.url)
    if res["status"] == "failed":
        raise HTTPException(400, f"Failed to acquire document from URL: {res.get('error')}")

    clean_text, title = res["clean_text"], res.get("title") or "Imported Case"
    if not is_land_case(text=clean_text, case_type=payload.case_type or ""):
        raise HTTPException(400, "NyaAi is currently designed for land cases. You can expect future updates to serve other case types.")

    struct = extract_legal_sections_structure(clean_text, kanoon_structure=res.get("kanoon_structure", {}))
    meta = res.get("metadata", {})
    court = meta.get("court") or ("Supreme Court of India" if "supreme court" in clean_text.lower() else "High Court" if "high court" in clean_text.lower() else "District Court")

    case = models.Case(
        title=title, court=court, judge=meta.get("judge") or meta.get("bench"),
        case_type=payload.case_type or "Land Dispute", legal_domain="Land & Property Law",
        facts=struct["facts"], issues="\n".join(struct["legal_issues"]) if isinstance(struct["legal_issues"], list) else str(struct["legal_issues"]),
        arguments={"petitioner": struct.get("petitioner_arguments", []), "respondent": struct.get("respondent_arguments", [])},
        evidence=[{"description": str(e)} for e in struct.get("evidence", [])],
        provisions=[{"act": l, "section": ""} for l in struct.get("laws", [])] + [{"act": "", "section": s} for s in struct.get("sections", [])],
        judgment=struct.get("decision", ""), procedural_history=f"Acquired from {payload.url}", status="indexed",
    )
    db.add(case); db.commit(); db.refresh(case)

    doc = models.Document(
        case_id=case.id, filename=f"url_import_{case.id}.txt", doc_type="judgment",
        source="url_acquisition", source_url=payload.url, document_status="ready",
        document_hash=res["document_hash"], full_text=clean_text,
    )
    db.add(doc); db.commit(); db.refresh(doc)
    index_document(db, doc)
    return case

@router.get("", response_model=list[schemas.CaseOut])
def list_cases(db: Session = Depends(get_db)):
    return db.query(models.Case).order_by(models.Case.created_at.desc()).all()

@router.get("/{case_id}", response_model=schemas.CaseOut)
def get_case(case_id: str, db: Session = Depends(get_db)):
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")
    return case

@router.patch("/{case_id}", response_model=schemas.CaseOut)
def update_case(case_id: str, payload: schemas.CaseUpdate, db: Session = Depends(get_db)):
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")
    data = payload.model_dump(exclude_unset=True)
    if "parties" in data and data["parties"] is not None:
        data["parties"] = [p if isinstance(p, dict) else p.model_dump() for p in data["parties"]]
    for k, v in data.items(): setattr(case, k, v)
    db.commit(); db.refresh(case)
    return case

@router.delete("/{case_id}")
def delete_case(case_id: str, db: Session = Depends(get_db)):
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")
    db.delete(case); db.commit()
    return {"deleted": True}

@router.get("/{case_id}/dashboard-stats")
def dashboard_stats(case_id: str, db: Session = Depends(get_db)):
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")
    return {
        "documents": db.query(models.Document).filter(models.Document.case_id == case_id).count(),
        "chunks": db.query(models.Chunk).filter(models.Chunk.case_id == case_id).count(),
        "analyses": db.query(models.Analysis).filter(models.Analysis.case_id == case_id).count(),
        "status": case.status,
    }

@router.get("/{case_id}/citation-graph")
def case_citation_graph(case_id: str, db: Session = Depends(get_db)):
    from ..services.citation_graph import get_citation_graph
    from ..services.retrieval import retrieve_historical_chunks
    graph = get_citation_graph(db)
    case = db.get(models.Case, case_id)
    target_id = None
    if case:
        q = " ".join(filter(None, [case.facts or "", case.issues or "", case.case_type or ""]))
        if q.strip():
            for r in retrieve_historical_chunks(db, q, top_k=5):
                if r.chunk.historical_case_id and r.chunk.historical_case_id in graph.graph:
                    target_id = r.chunk.historical_case_id; break
    if not target_id:
        top_hist = db.query(models.HistoricalCase).first()
        target_id = top_hist.id if top_hist else case_id
    return graph.get_case_citation_subgraph(target_id, depth=2)

@router.get("/{case_id}/related-precedents")
def get_related_precedents(case_id: str, db: Session = Depends(get_db)):
    from ..services.citation_graph import get_citation_graph
    from ..services.retrieval import retrieve_historical_chunks
    from ..services.reranker import rerank_candidates
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")

    q = " ".join(filter(None, [case.case_type or "", case.facts or "", case.charges_issues or case.issues or "", case.arguments_text or ""]))
    if not q.strip():
        return []

    retrieved = retrieve_historical_chunks(db, q, top_k=25, per_case_cap=3)
    if not retrieved:
        return []

    # Rerank for genuine relevance, keep top 15 candidates
    reranked = rerank_candidates(db, case, retrieved, top_k=15)
    score_map = {r.historical_case.id: (round(r.total_score, 3), r.explanation, r.to_dict().get("score_breakdown")) for r in reranked}

    # Citation graph expansion (top authorities directly linked)
    retrieved_hc_ids = list(dict.fromkeys(r.chunk.historical_case_id for r in retrieved if r.chunk.historical_case_id))
    graph = get_citation_graph(db)
    graph_ids = []
    target_id = next((hid for hid in retrieved_hc_ids if hid in graph.graph), None)
    if target_id:
        sub = graph.get_case_citation_subgraph(target_id, depth=1)
        graph_ids = [n["id"] for n in sub.get("nodes", []) if n["id"] != target_id]

    priority_ids = [r.historical_case.id for r in reranked]
    combined_ids = list(dict.fromkeys(priority_ids + graph_ids))

    cases = db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(combined_ids)).all()
    case_map = {c.id: c for c in cases}

    result = []
    seen_names = set()
    current_title_lower = (case.title or "").strip().lower()

    for cid in combined_ids:
        c = case_map.get(cid)
        if not c:
            continue
        # Exclude current case if names or IDs match
        norm_name = c.case_name.strip().lower()
        if c.id == case.id or norm_name == current_title_lower:
            continue
        # Deduplicate identical or near-identical case titles
        if norm_name in seen_names:
            continue
        seen_names.add(norm_name)

        score_info = score_map.get(cid)
        if score_info:
            score, why, breakdown = score_info
        else:
            score, why, breakdown = None, f"Connected authority cited within precedent network.", None

        result.append({
            "id": c.id, "case_name": c.case_name, "court": c.court, "year": c.year,
            "citation": c.citation, "outcome": c.outcome, "summary": c.summary,
            "relevance_score": score,
            "why_relevant": why or f"Relevant precedent for {case.case_type or 'this legal issue'}",
            "score_breakdown": breakdown,
        })
        if len(result) >= 15:
            break

    return result
