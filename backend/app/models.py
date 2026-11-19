import uuid, datetime as dt
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base

def gen_id() -> str:
    return uuid.uuid4().hex[:12]

def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

class Case(Base):
    __tablename__ = "cases"
    id = Column(String, primary_key=True, default=gen_id)
    title = Column(String, nullable=False)
    case_number = Column(String, nullable=True)
    court = Column(String, nullable=True)
    jurisdiction = Column(String, nullable=True)
    judge = Column(String, nullable=True)
    date_of_judgment = Column(String, nullable=True)
    date_of_filing = Column(String, nullable=True)
    parties = Column(JSON, default=list)
    case_type = Column(String, nullable=True)
    legal_domain = Column(String, nullable=True)
    provisions = Column(JSON, default=list)
    facts = Column(Text, default="")
    issues = Column(Text, default="")
    arguments = Column(JSON, default=dict)
    evidence = Column(JSON, default=list)
    procedural_history = Column(Text, default="")
    judgment = Column(Text, default="")
    timeline = Column(JSON, default=list)
    status = Column(String, default="draft")
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
                if isinstance(args, list): parts.extend([f"{role}: {a}" for a in args])
                elif args: parts.append(f"{role}: {args}")
            return "\n".join(parts)
        return self.arguments if isinstance(self.arguments, str) else ""

    @arguments_text.setter
    def arguments_text(self, val: str):
        self.arguments = {"general": [line.strip() for line in (val or "").splitlines() if line.strip()]}

    @property
    def evidence_text(self) -> str:
        if isinstance(self.evidence, list):
            items = []
            for e in self.evidence:
                items.append(e.get("description") or e.get("name") or str(e) if isinstance(e, dict) else str(e) if e else "")
            return "\n".join([i for i in items if i])
        return self.evidence if isinstance(self.evidence, str) else ""

    @evidence_text.setter
    def evidence_text(self, val: str):
        self.evidence = [{"description": line.strip()} for line in (val or "").splitlines() if line.strip()]

class HistoricalCase(Base):
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
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=gen_id)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    historical_case_id = Column(String, ForeignKey("historical_cases.id"), nullable=True)
    filename = Column(String, nullable=False)
    doc_type = Column(String, default="general")
    source = Column(String, default="upload")
    source_url = Column(String, nullable=True)
    document_status = Column(String, default="ready")
    document_hash = Column(String, nullable=True)
    retrieved_at = Column(DateTime, nullable=True)
    upload_date = Column(DateTime, default=now)
    full_text = Column(Text, default="")
    num_chunks = Column(Integer, default=0)

    case = relationship("Case", back_populates="documents")
    historical_case = relationship("HistoricalCase", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    historical_case_id = Column(String, ForeignKey("historical_cases.id"), nullable=True)
    chunk_index = Column(Integer, default=0)
    section_type = Column(String, default="body")
    page_number = Column(Integer, nullable=True)
    char_start = Column(Integer, default=0)
    char_end = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)

    document = relationship("Document", back_populates="chunks")

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String, primary_key=True, default=gen_id)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    provider = Column(String, default="mock")
    content = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)

    case = relationship("Case", back_populates="analyses")
    citations = relationship("Citation", back_populates="analysis", cascade="all, delete-orphan")

class Citation(Base):
    __tablename__ = "citations"
    id = Column(String, primary_key=True, default=gen_id)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    ref_key = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    chunk_id = Column(String, ForeignKey("chunks.id"), nullable=True)
    historical_case_id = Column(String, nullable=True)
    case_name = Column(String, nullable=True)
    document_filename = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    excerpt = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)

    analysis = relationship("Analysis", back_populates="citations")
