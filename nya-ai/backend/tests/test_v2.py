import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base
from app import models
from app.services.chunking import chunk_text
from app.services.indexing import index_document
from app.services.retrieval import retrieve_historical_chunks
from app.services.case_extraction import extract_case_profile
from app.services.workflow import process_document_workflow
from app.main import app
import os

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # We must ensure we're using mock models for testing
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["LLM_PROVIDER"] = "mock"
    
    yield db
    
    db.close()
    Base.metadata.drop_all(bind=engine)

def test_chunking_and_offsets():
    text = "This is a paragraph.\n\nThis is another paragraph."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "This is a paragraph." in chunks[0].text
    assert chunks[0].char_start == 0

def test_case_extraction_mock(db):
    case = models.Case(title="Test Case")
    db.add(case)
    db.commit()
    
    extract_case_profile(db, case.id, "Some test document text.")
    
    db.refresh(case)
    assert case.title == "Extracted Case (Mock)"
    assert case.case_number == "MOCK-1234"
    assert "mock extracted fact" in case.facts

def test_hybrid_retrieval(db):
    # Create historical case
    hc = models.HistoricalCase(case_name="Precedent A")
    db.add(hc)
    db.commit()
    
    # Add chunks
    chunk = models.Chunk(
        document_id="fake_doc",
        historical_case_id=hc.id,
        text="The quick brown fox jumps over the lazy dog.",
        embedding=[0.1] * 512  # fake embedding
    )
    db.add(chunk)
    db.commit()
    
    # query that matches keywords >= 4 chars
    results = retrieve_historical_chunks(db, "quick jumps", top_k=5)
    assert len(results) > 0
    assert results[0].chunk.historical_case_id == hc.id
    assert results[0].keyword_score > 0

def test_background_workflow(db, monkeypatch):
    import app.services.workflow
    # mock SessionLocal to use the test db
    monkeypatch.setattr(app.services.workflow, "SessionLocal", lambda: db)
    
    case = models.Case(title="Workflow Case")
    db.add(case)
    db.commit()
    
    doc = models.Document(
        case_id=case.id,
        filename="test.txt",
        doc_type="general",
        full_text="Test document text for workflow."
    )
    db.add(doc)
    db.commit()
    
    process_document_workflow(case.id, doc.id)
    
    db.refresh(case)
    if case.status == "error":
        print(case.error_message)
    assert case.status == "analyzed"
    assert case.num_chunks_indexed() > 0
    
    analyses = db.query(models.Analysis).filter_by(case_id=case.id).all()
    assert len(analyses) > 0
