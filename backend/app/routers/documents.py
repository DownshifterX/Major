from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.text_extraction import extract_text, UnsupportedFileType
from ..services.indexing import index_document
from ..services.workflow import process_document_workflow

router = APIRouter(prefix="/cases", tags=["documents"])

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15MB safety cap


@router.post("/{case_id}/documents", response_model=schemas.DocumentOut)
async def upload_document(
    case_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    db: Session = Depends(get_db),
):
    case = db.get(models.Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, "File too large (max 15MB in this PoC)")
    if not raw:
        raise HTTPException(400, "Empty file")

    try:
        text = extract_text(file.filename, raw)
    except UnsupportedFileType as e:
        raise HTTPException(400, str(e))

    if not text.strip():
        raise HTTPException(422, "No extractable text found in this document")

    from ..services.legal_structure import is_land_case
    if not is_land_case(text=text, case_type=case.case_type or ""):
        raise HTTPException(
            400,
            "NyaAi is currently designed for land cases. You can expect future updates to serve other case types."
        )

    document = models.Document(
        case_id=case_id,
        filename=file.filename,
        doc_type=doc_type,
        source="upload",
        full_text=text,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(process_document_workflow, case_id, document.id)

    case.status = "processing"
    db.commit()
    return document


@router.get("/{case_id}/documents", response_model=list[schemas.DocumentOut])
def list_documents(case_id: str, db: Session = Depends(get_db)):
    return db.query(models.Document).filter(models.Document.case_id == case_id).all()


@router.delete("/{case_id}/documents/{document_id}")
def delete_document(case_id: str, document_id: str, db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(
        models.Document.id == document_id, models.Document.case_id == case_id
    ).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()
    return {"deleted": True}


@router.post("/{case_id}/index")
def reindex_case(case_id: str, db: Session = Depends(get_db)):
    case = db.get(models.Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    docs = db.query(models.Document).filter(models.Document.case_id == case_id).all()
    total = 0
    for d in docs:
        total += index_document(db, d)
    case.status = "indexed" if docs else "draft"
    db.commit()
    return {"documents_indexed": len(docs), "chunks_created": total}
