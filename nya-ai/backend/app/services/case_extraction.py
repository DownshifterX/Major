import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from .. import models
from .llm_provider import get_llm_provider

EXTRACTION_PROMPT = """You are an expert legal AI assistant. Your task is to extract a comprehensive structured profile from the provided legal case document.
You must strictly return valid JSON matching the schema below. Do not include any markdown formatting, preamble, or commentary.

SCHEMA:
{
  "title": "String - Case title or name",
  "case_number": "String",
  "court": "String",
  "jurisdiction": "String",
  "judge": "String",
  "date_of_judgment": "String",
  "date_of_filing": "String",
  "case_type": "String",
  "legal_domain": "String",
  "parties": [{"role": "petitioner/respondent/etc", "name": "String"}],
  "facts": "String - Chronological summary of material facts",
  "issues": "String - Legal and factual issues before the court",
  "arguments": {
     "petitioner": ["String"],
     "respondent": ["String"]
  },
  "evidence": [{"description": "String", "type": "documentary/witness/etc"}],
  "provisions": [{"act": "String", "section": "String"}],
  "procedural_history": "String",
  "judgment": "String - Final decision and orders",
  "timeline": [{"date": "String", "event": "String"}]
}

If any information is not present in the text, omit it or leave it empty. DO NOT hallucinate facts.

DOCUMENT TEXT:
{text}
"""

def extract_case_profile(db: Session, case_id: str, document_text: str):
    """Automatically extracts structured case information from the uploaded document."""
    case = db.query(models.Case).get(case_id)
    if not case:
        return
    
    provider = get_llm_provider()
    
    # Truncate text for context window
    text_to_process = document_text[:40000] 
    
    prompt = EXTRACTION_PROMPT.replace("{text}", text_to_process)
    
    # Handle mock explicitly if mock provider is active
    if provider.__class__.__name__ == "MockLLMProvider":
        result = provider.generate_structured(prompt, context={"text": text_to_process}, analysis_type="case_extraction")
    else:
        result = provider.generate_structured(prompt, context={}, analysis_type="case_extraction")
    
    if "error" in result:
        case.error_message = result["error"]
        case.status = "error"
        db.commit()
        return

    # Update case with extracted fields
    case.title = result.get("title") or case.title or "Untitled Case"
    case.case_number = result.get("case_number", "")
    case.court = result.get("court", "")
    case.jurisdiction = result.get("jurisdiction", "")
    case.judge = result.get("judge", "")
    case.date_of_judgment = result.get("date_of_judgment", "")
    case.date_of_filing = result.get("date_of_filing", "")
    case.case_type = result.get("case_type", "")
    case.legal_domain = result.get("legal_domain", "")
    case.parties = result.get("parties", [])
    case.facts = result.get("facts", "")
    case.issues = result.get("issues", "")
    case.arguments = result.get("arguments", {})
    case.evidence = result.get("evidence", [])
    case.provisions = result.get("provisions", [])
    case.procedural_history = result.get("procedural_history", "")
    case.judgment = result.get("judgment", "")
    case.timeline = result.get("timeline", [])
    
    db.commit()
