import traceback
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
        document = db.get(models.Document, document_id)
        
        if not case or not document:
            return
            
        case.status = "processing"
        db.commit()
        
        # 1. Automatic Case Extraction
        extract_case_profile(db, case_id, document.full_text)
        
        # 2. Embedding / Indexing
        index_document(db, document)
        
        # 3. Automatic Analysis (Similar cases, provisions, gaps, etc.)
        svc.run_similar_cases(db, case)
        svc.run_legal_provisions(db, case)
        svc.run_prosecution_defence(db, case)
        svc.run_evidence_gaps(db, case)
        svc.run_contradictions(db, case)
        svc.run_brief(db, case)
        
        case.status = "analyzed"
        db.commit()
    except Exception as e:
        db.rollback()
        case = db.get(models.Case, case_id)
        if case:
            case.status = "error"
            case.error_message = f"Workflow failed: {str(e)}\n{traceback.format_exc()}"
            db.commit()
    finally:
        db.close()
