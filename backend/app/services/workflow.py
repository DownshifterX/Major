from sqlalchemy.orm import Session
from ..database import SessionLocal
from .. import models
from .case_extraction import extract_case_profile
from .indexing import index_document
from . import analysis_service as svc

def process_document_workflow(case_id: str, document_id: str):
    db: Session = SessionLocal()
    try:
        case = db.get(models.Case, case_id)
        doc = db.get(models.Document, document_id)
        if not case or not doc: return

        case.status = "processing"; db.commit()
        extract_case_profile(db, case_id, doc.full_text)
        index_document(db, doc)

        for fn in [svc.run_similar_cases, svc.run_legal_provisions, svc.run_prosecution_defence,
                   svc.run_evidence_gaps, svc.run_contradictions, svc.run_brief]:
            fn(db, case)

        case.status = "analyzed"; db.commit()
    except Exception as e:
        db.rollback()
        case = db.get(models.Case, case_id)
        if case:
            case.status = "error"
            case.error_message = f"Workflow failed: {e}"
            db.commit()
    finally:
        db.close()
