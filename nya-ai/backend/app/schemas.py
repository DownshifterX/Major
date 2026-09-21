from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PartyIn(BaseModel):
    role: str
    name: str


class CaseCreate(BaseModel):
    title: str
    case_number: Optional[str] = None
    court: Optional[str] = None
    jurisdiction: Optional[str] = None
    judge: Optional[str] = None
    date_of_judgment: Optional[str] = None
    date_of_filing: Optional[str] = None
    case_type: Optional[str] = None
    legal_domain: Optional[str] = None
    parties: Optional[List[PartyIn]] = []
    provisions: Optional[List[Dict[str, Any]]] = []
    facts: Optional[str] = ""
    issues: Optional[str] = ""
    charges_issues: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = {}
    arguments_text: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = []
    evidence_text: Optional[str] = None
    procedural_history: Optional[str] = ""
    judgment: Optional[str] = ""
    timeline: Optional[List[Dict[str, Any]]] = []


class UrlImportRequest(BaseModel):
    url: str
    case_type: Optional[str] = "General"


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    case_number: Optional[str] = None
    court: Optional[str] = None
    jurisdiction: Optional[str] = None
    judge: Optional[str] = None
    date_of_judgment: Optional[str] = None
    date_of_filing: Optional[str] = None
    case_type: Optional[str] = None
    legal_domain: Optional[str] = None
    parties: Optional[List[PartyIn]] = None
    provisions: Optional[List[Dict[str, Any]]] = None
    facts: Optional[str] = None
    issues: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    procedural_history: Optional[str] = None
    judgment: Optional[str] = None
    timeline: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    error_message: Optional[str] = None


class CaseOut(BaseModel):
    id: str
    title: str
    case_number: Optional[str]
    court: Optional[str]
    jurisdiction: Optional[str]
    judge: Optional[str]
    date_of_judgment: Optional[str]
    date_of_filing: Optional[str]
    case_type: Optional[str]
    legal_domain: Optional[str]
    parties: Optional[List[Dict[str, Any]]]
    provisions: Optional[List[Dict[str, Any]]]
    facts: str
    issues: str
    arguments: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    procedural_history: str
    judgment: str
    timeline: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str
    source: str
    num_chunks: int

    class Config:
        from_attributes = True


class CompareRequest(BaseModel):
    historical_case_id: str
