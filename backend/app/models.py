import uuid
import datetime as dt
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, JSON
)
from sqlalchemy.orm import relationship
from .database import Base


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> dt.datetime:
    return dt.datetime.utcnow()


class Case(Base):
    """A current, live case being worked on by the user."""
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=gen_id)
    title = Column(String, nullable=False)
    case_number = Column(String, nullable=True)
    court = Column(String, nullable=True)
    jurisdiction = Column(String, nullable=True)
    judge = Column(String, nullable=True)
    date_of_judgment = Column(String, nullable=True)
    date_of_filing = Column(String, nullable=True)
    parties = Column(JSON, default=list)                 # [{"role": "...", "name": "..."}]
    case_type = Column(String, nullable=True)          # e.g. Criminal, Civil, Contract
    legal_domain = Column(String, nullable=True)
    provisions = Column(JSON, default=list)              # Extracted statutes/sections

    facts = Column(Text, default="")
    issues = Column(Text, default="")                    # Replaces charges_issues
    arguments = Column(JSON, default=dict)               # e.g., {"petitioner": [...], "respondent": [...]}
    evidence = Column(JSON, default=list)                # Structured evidence list
    procedural_history = Column(Text, default="")
    judgment = Column(Text, default="")
    timeline = Column(JSON, default=list)                 # [{"date": "...", "event": "..."}]
    
    status = Column(String, default="draft")             # draft | processing | indexed | analyzed | error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)

    documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="case", cascade="all, delete-orphan")

    def num_chunks_indexed(self) -> int:
        return sum(doc.num_chunks or 0 for doc in self.documents)

    @property
    def charges_issues(self) -> str:
        return self.issues or ""

    @charges_issues.setter
    def charges_issues(self, val: str):
        self.issues = val or ""

    @property
    def arguments_text(self) -> str:
        if isinstance(self.arguments, dict):
            parts = []
            for role, args in self.arguments.items():
                if isinstance(args, list):
                    parts.extend([f"{role}: {a}" for a in args])
                elif args:
                    parts.append(f"{role}: {args}")
            return "\n".join(parts)
        elif isinstance(self.arguments, str):
            return self.arguments
        return ""

    @arguments_text.setter
    def arguments_text(self, val: str):
        self.arguments = {"general": [line.strip() for line in (val or "").splitlines() if line.strip()]}

    @property
    def evidence_text(self) -> str:
        if isinstance(self.evidence, list):
            items = []
            for e in self.evidence:
                if isinstance(e, dict):
                    desc = e.get("description") or e.get("name") or str(e)
                    items.append(desc)
                elif e:
                    items.append(str(e))
            return "\n".join(items)
        elif isinstance(self.evidence, str):
            return self.evidence
        return ""

    @evidence_text.setter
    def evidence_text(self, val: str):
        self.evidence = [{"description": line.strip()} for line in (val or "").splitlines() if line.strip()]


class HistoricalCase(Base):
    """An entry in the historical precedent corpus used for retrieval."""
    __tablename__ = "historical_cases"

    id = Column(String, primary_key=True, default=gen_id)
    case_name = Column(String, nullable=False)
    court = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    citation = Column(String, nullable=True)
    case_type = Column(String, nullable=True)
    outcome = Column(Text, default="")
    summary = Column(Text, default="")
    doc_url = Column(String, nullable=True)
    cites_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    court_normalized = Column(String, nullable=True)
    legal_sections = Column(JSON, default=list)
    laws = Column(JSON, default=list)
    is_synthetic = Column(Boolean, default=False)
    source_note = Column(String, default="")
    created_at = Column(DateTime, default=now)

    documents = relationship("Document", back_populates="historical_case", cascade="all, delete-orphan")


class Document(Base):
    """A source document — belongs to either a live Case or a HistoricalCase."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_id)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    historical_case_id = Column(String, ForeignKey("historical_cases.id"), nullable=True)
    filename = Column(String, nullable=False)
    doc_type = Column(String, default="general")   # fir/complaint/statement/judgment/evidence/other
    source = Column(String, default="upload")        # upload | seed_corpus | url_acquisition
    source_url = Column(String, nullable=True)
    document_status = Column(String, default="ready") # ready | acquiring | failed | cached
    document_hash = Column(String, nullable=True)
    retrieved_at = Column(DateTime, nullable=True)
    upload_date = Column(DateTime, default=now)
    full_text = Column(Text, default="")
    num_chunks = Column(Integer, default=0)

    case = relationship("Case", back_populates="documents")
    historical_case = relationship("HistoricalCase", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A retrieval unit: a slice of a document's text plus its embedding vector."""
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)                 # denormalized
    historical_case_id = Column(String, ForeignKey("historical_cases.id"), nullable=True)
    chunk_index = Column(Integer, default=0)
    section_type = Column(String, default="body")   # facts | issues | arguments | evidence | laws | reasoning | decision | body
    page_number = Column(Integer, nullable=True)
    char_start = Column(Integer, default=0)
    char_end = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)   # list[float]
    created_at = Column(DateTime, default=now)

    document = relationship("Document", back_populates="chunks")


class Analysis(Base):
    """A stored result of a RAG analysis run (one row per analysis_type per case run)."""
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=gen_id)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    # similar_cases | legal_provisions | prosecution_defence | evidence_gaps |
    # contradictions | comparison | brief
    provider = Column(String, default="mock")
    content = Column(JSON, default=dict)       # structured output
    created_at = Column(DateTime, default=now)

    case = relationship("Case", back_populates="analyses")
    citations = relationship("Citation", back_populates="analysis", cascade="all, delete-orphan")


class Citation(Base):
    """First-class, structured provenance object linking a claim to a real source chunk."""
    __tablename__ = "citations"

    id = Column(String, primary_key=True, default=gen_id)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    ref_key = Column(String, nullable=False)          # e.g. "S1", referenced inline by the claim
    source_type = Column(String, nullable=False)      # historical_case | current_document
    chunk_id = Column(String, ForeignKey("chunks.id"), nullable=True)
    historical_case_id = Column(String, nullable=True)
    case_name = Column(String, nullable=True)
    document_filename = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    excerpt = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)

    analysis = relationship("Analysis", back_populates="citations")
