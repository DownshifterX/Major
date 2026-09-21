"""Nya AI — Comprehensive API Integration Test Suite.

Tests the full FastAPI layer end-to-end using TestClient + in-memory SQLite.
"""
import os
import sys
import pytest

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "mock")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app import models
from app.main import app

from sqlalchemy.pool import StaticPool

SQLALCHEMY_TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def db_session():
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture()
def seeded_corpus(db_session):
    hc1 = models.HistoricalCase(
        case_name="State vs Ramesh Kumar — Murder",
        court="Supreme Court of India",
        year=2018,
        citation="2018 SCC 123",
        cited_by_count=45,
        summary="Murder conviction under Section 302 IPC upheld. Eyewitness testimony relied upon.",
        is_synthetic=False,
    )
    hc2 = models.HistoricalCase(
        case_name="Singh vs State — Anticipatory Bail",
        court="High Court of Delhi",
        year=2021,
        citation="2021 ILR 456",
        cited_by_count=12,
        summary="Anticipatory bail under Section 438 CrPC granted. Nature of offence considered.",
        is_synthetic=False,
    )
    db_session.add_all([hc1, hc2])
    db_session.commit()

    for hc in [hc1, hc2]:
        doc = models.Document(
            historical_case_id=hc.id,
            filename=f"judgment_{hc.id}.txt",
            doc_type="judgment",
            source="seed_corpus",
            full_text=hc.summary * 12,
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        for i, snippet in enumerate([hc.summary[:200], hc.summary]):
            db_session.add(models.Chunk(
                document_id=doc.id,
                historical_case_id=hc.id,
                chunk_index=i,
                text=snippet,
                embedding=[0.1 * (i + 1)] * 512,
            ))
        doc.num_chunks = 2
        db_session.commit()

    return hc1, hc2


# ===== Health =====

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ===== Case CRUD =====

def test_create_case(client):
    r = client.post("/cases", json={
        "title": "Criminal Case Test",
        "case_type": "Criminal",
        "facts": "Accused found with stolen goods.",
        "charges_issues": "Section 379 IPC theft",
        "arguments_text": "Prosecution: caught red-handed\nDefence: alibi",
        "evidence_text": "CCTV footage\nPolice statement",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Criminal Case Test"
    assert data["status"] == "draft"


def test_list_cases(client):
    client.post("/cases", json={"title": "A"})
    client.post("/cases", json={"title": "B"})
    r = client.get("/cases")
    assert r.status_code == 200
    assert len(r.json()) >= 2


def test_get_case(client):
    cid = client.post("/cases", json={"title": "Get Me"}).json()["id"]
    r = client.get(f"/cases/{cid}")
    assert r.status_code == 200
    assert r.json()["title"] == "Get Me"


def test_get_case_not_found(client):
    assert client.get("/cases/doesnotexist").status_code == 404


def test_update_case(client):
    cid = client.post("/cases", json={"title": "Old"}).json()["id"]
    r = client.patch(f"/cases/{cid}", json={"title": "New", "status": "indexed"})
    assert r.status_code == 200
    assert r.json()["title"] == "New"


def test_delete_case(client):
    cid = client.post("/cases", json={"title": "To Delete"}).json()["id"]
    assert client.delete(f"/cases/{cid}").status_code == 200
    assert client.get(f"/cases/{cid}").status_code == 404


# ===== Dashboard Stats =====

def test_dashboard_stats(client):
    cid = client.post("/cases", json={"title": "Dash"}).json()["id"]
    r = client.get(f"/cases/{cid}/dashboard-stats")
    assert r.status_code == 200
    data = r.json()
    assert all(k in data for k in ["documents", "chunks", "analyses", "status"])


# ===== Analysis Pipeline =====

def test_full_analysis_pipeline(client, seeded_corpus):
    cid = client.post("/cases", json={
        "title": "Full Analysis Test",
        "case_type": "Criminal",
        "facts": "Murder under Section 302 IPC with eyewitness evidence.",
        "charges_issues": "Section 302 IPC murder",
        "arguments_text": "Prosecution: evidence conclusive\nDefence: alibi",
        "evidence_text": "Eyewitness\nForensic report",
    }).json()["id"]

    r = client.post(f"/cases/{cid}/analyze", json={})
    assert r.status_code == 200
    data = r.json()
    for key in ["similar_cases", "legal_provisions", "prosecution_defence", "evidence_gaps", "contradictions"]:
        assert key in data, f"Missing key: {key}"


def test_similar_cases_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={
        "title": "Bail Case",
        "facts": "Section 438 CrPC anticipatory bail application",
        "charges_issues": "Section 438 CrPC",
    }).json()["id"]
    r = client.get(f"/cases/{cid}/similar-cases")
    assert r.status_code == 200
    assert "citations" in r.json()


def test_legal_provisions_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={"title": "Provisions Test", "facts": "Section 302 IPC"}).json()["id"]
    r = client.get(f"/cases/{cid}/legal-provisions")
    assert r.status_code == 200
    assert "legal_provisions" in r.json()


def test_prosecution_defence_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={
        "title": "P&D Test",
        "arguments_text": "Prosecution: evidence is clear\nDefence: no motive",
    }).json()["id"]
    r = client.get(f"/cases/{cid}/prosecution-defence")
    assert r.status_code == 200
    data = r.json()
    assert "prosecution" in data
    assert "defence" in data


def test_evidence_gaps_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={
        "title": "Evidence Test",
        "evidence_text": "CCTV footage\nForensic report",
    }).json()["id"]
    r = client.get(f"/cases/{cid}/evidence-gaps")
    assert r.status_code == 200
    assert "evidence_present" in r.json()


def test_contradictions_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={"title": "Contradiction Test", "facts": "some facts"}).json()["id"]
    r = client.get(f"/cases/{cid}/contradictions")
    assert r.status_code == 200
    assert "contradictions" in r.json()


def test_brief_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={"title": "Brief Test", "facts": "Brief case facts"}).json()["id"]
    r = client.get(f"/cases/{cid}/brief")
    assert r.status_code == 200
    data = r.json()
    assert "case_overview" in data or "key_facts" in data


def test_compare_endpoint(client, seeded_corpus):
    hc1, _ = seeded_corpus
    cid = client.post("/cases", json={
        "title": "Compare Source",
        "facts": "Murder eyewitness Section 302",
    }).json()["id"]
    r = client.post(f"/cases/{cid}/compare", json={"historical_case_id": hc1.id})
    assert r.status_code == 200
    data = r.json()
    assert "historical_case" in data


# ===== Citation Graph =====

def test_citation_graph(client, seeded_corpus):
    cid = client.post("/cases", json={"title": "Graph Test"}).json()["id"]
    r = client.get(f"/cases/{cid}/citation-graph")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "links" in data
    assert "metrics" in data


# ===== Search =====

def test_api_search_with_corpus(client, seeded_corpus):
    r = client.get("/api/search?q=Section+302+IPC+murder&limit=5")
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)
    for item in results:
        assert "case_name" in item
        assert "hybrid_score" in item
        assert "excerpt" in item


def test_api_search_empty_corpus(client):
    r = client.get("/api/search?q=test&limit=5")
    assert r.status_code == 200
    assert r.json() == []


# ===== Reranked Precedents =====

def test_reranked_precedents_endpoint(client, seeded_corpus):
    cid = client.post("/cases", json={
        "title": "Rerank Test",
        "facts": "Murder Section 302 IPC conviction eyewitness",
        "issues": "Section 302 IPC murder",
    }).json()["id"]
    r = client.get(f"/api/cases/{cid}/reranked-precedents?top_k=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "total_score" in item
        assert "score_breakdown" in item
        bd = item["score_breakdown"]
        for key in ["semantic", "lexical", "statute_overlap", "citation_authority", "court_weight"]:
            assert key in bd, f"Missing score breakdown key: {key}"


def test_reranked_precedents_not_found(client):
    r = client.get("/api/cases/doesnotexist/reranked-precedents")
    assert r.status_code == 404


# ===== Historical Corpus =====

def test_list_historical_cases(client, seeded_corpus):
    r = client.get("/historical-cases")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    names = [d["case_name"] for d in data]
    assert any("Ramesh Kumar" in n for n in names)


def test_get_historical_case(client, seeded_corpus):
    hc1, _ = seeded_corpus
    r = client.get(f"/historical-cases/{hc1.id}")
    assert r.status_code == 200
    data = r.json()
    assert data["case_name"] == hc1.case_name
    assert "documents" in data


def test_get_historical_case_not_found(client):
    assert client.get("/historical-cases/invalid123").status_code == 404
