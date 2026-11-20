from sqlalchemy.orm import Session
from .. import models
from .llm_provider import get_llm_provider

EXTRACTION_PROMPT = """You are an expert legal AI assistant. Extract a structured profile from the case document.
Strictly return valid JSON:
{
  "title": "String", "case_number": "String", "court": "String", "jurisdiction": "String",
  "judge": "String", "date_of_judgment": "String", "date_of_filing": "String", "case_type": "String",
  "legal_domain": "String", "parties": [{"role": "String", "name": "String"}],
  "facts": "String", "issues": "String", "arguments": {"petitioner": ["String"], "respondent": ["String"]},
  "evidence": [{"description": "String"}], "provisions": [{"act": "String", "section": "String"}],
  "procedural_history": "String", "judgment": "String", "timeline": [{"date": "String", "event": "String"}]
}
DOCUMENT:
{text}"""

def extract_case_profile(db: Session, case_id: str, document_text: str):
    case = db.get(models.Case, case_id)
    if not case: return
    provider = get_llm_provider()
    txt = (document_text or "")[:40000]
    prompt = EXTRACTION_PROMPT.replace("{text}", txt)
    res = provider.generate_structured(prompt, context={"text": txt}, analysis_type="case_extraction")

    if "error" in res:
        case.error_message, case.status = res["error"], "error"
        db.commit(); return

    for field in ["title", "case_number", "court", "jurisdiction", "judge", "date_of_judgment",
                  "date_of_filing", "case_type", "legal_domain", "parties", "facts", "issues",
                  "arguments", "evidence", "provisions", "procedural_history", "judgment", "timeline"]:
        val = res.get(field)
        if val is not None:
            setattr(case, field, val)
    db.commit()
