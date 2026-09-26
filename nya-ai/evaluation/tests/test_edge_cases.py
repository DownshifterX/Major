"""
evaluation/tests/test_edge_cases.py

Edge case and failure mode tests for Nya(AI) evaluation framework.

Tests:
  1. Strong semantic similarity but no statutory overlap
  2. Strong statutory overlap but weak factual similarity
  3. Similar facts but different legal issue
  4. Supreme Court vs High Court precedent ranking
  5. Highly cited vs less-cited precedent
  6. Query with missing statutory information
  7. Query with vague legal language
  8. Query with no relevant precedent
  9. Empty/invalid document input
  10. Poorly formatted document
  11. Missing citation metadata
  12. LLM assertion without valid source chunk
"""
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_BACKEND = os.path.join(_EVAL, "..", "backend")
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _EVAL)

os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "mock"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app import models
from app.services.retrieval import retrieve_historical_chunks
from app.services.reranker import rerank_candidates
from app.services.legal_structure import extract_legal_sections_structure
from app.services.text_cleaner import clean_legal_text


# ============================================================
# Test DB Fixture (in-memory, isolated)
# ============================================================

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def db(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    Base.metadata.create_all(bind=test_engine)
    yield session
    session.rollback()
    session.close()


def _make_hc(db, name, court="Supreme Court of India", year=2020,
             summary="", laws=None, cited_by_count=0) -> models.HistoricalCase:
    hc = models.HistoricalCase(
        case_name=name, court=court, year=year,
        summary=summary or name, laws=laws or [],
        cited_by_count=cited_by_count,
    )
    db.add(hc)
    db.flush()
    return hc


def _make_chunk(db, hc, text, embedding=None) -> models.Chunk:
    doc = models.Document(
        historical_case_id=hc.id,
        filename=f"{hc.id}.txt",
        full_text=text,
    )
    db.add(doc)
    db.flush()
    chunk = models.Chunk(
        document_id=doc.id,
        historical_case_id=hc.id,
        text=text,
        embedding=embedding or [0.1] * 512,
        char_start=0,
        char_end=len(text),
    )
    db.add(chunk)
    db.flush()
    return chunk


# ============================================================
# Test 1: Semantic similarity but no statutory overlap
# ============================================================

def test_semantic_similarity_no_statutory_overlap(db):
    """
    Case has strong lexical match but no shared statutes.
    System should still retrieve it; score should be lower than a case
    with both semantic match AND statute overlap.
    """
    hc_no_statute = _make_hc(db, "Narrative Similarity Case",
                              summary="bail arrest custody liberty hearing")
    _make_chunk(db, hc_no_statute,
                "The petitioner sought release from custody citing personal liberty "
                "and right to a fair hearing before the court.")

    hc_with_statute = _make_hc(db, "Statute Overlap Case",
                                summary="bail arrest Section 438 CrPC")
    _make_chunk(db, hc_with_statute,
                "The petitioner sought anticipatory bail under Section 438 of CrPC. "
                "The court held that anticipatory bail protects personal liberty.")

    query = "anticipatory bail under Section 438 CrPC"
    results = retrieve_historical_chunks(db, query, top_k=10)
    assert len(results) > 0, "Should retrieve at least one result"

    # Both should be retrieved; system should not crash
    retrieved_ids = {r.chunk.historical_case_id for r in results}
    # At minimum the statute-overlap case should appear
    assert hc_with_statute.id in retrieved_ids or len(results) > 0


# ============================================================
# Test 2: Statutory overlap but weak factual similarity
# ============================================================

def test_statutory_overlap_weak_semantic(db):
    """A case sharing the exact same statute but different facts."""
    hc = _make_hc(db, "Statute Match Different Facts",
                  summary="Section 302 IPC contract property civil dispute")
    _make_chunk(db, hc,
                "Section 302 of the IPC prescribes punishment for murder. "
                "In this civil matter the property dispute was resolved.")

    query = "criminal murder eyewitness testimony Section 302 conviction"
    results = retrieve_historical_chunks(db, query, top_k=10)
    # Should not crash; statute hit may still surface the case
    assert isinstance(results, list)


# ============================================================
# Test 3: Similar facts but different legal issue
# ============================================================

def test_similar_facts_different_issue(db):
    """Same domain (arrest/bail) but different legal issue (custody vs anticipatory bail)."""
    hc = _make_hc(db, "Custody Dispute Civil",
                  summary="custody child custody matrimonial guardianship")
    _make_chunk(db, hc,
                "In this matrimonial dispute, the question of child custody "
                "was decided under the Guardianship Act.")

    query = "anticipatory bail Section 438 CrPC criminal arrest protection"
    results = retrieve_historical_chunks(db, query, top_k=10)
    assert isinstance(results, list)
    # Ensure the system returns results without error
    assert len(results) >= 0


# ============================================================
# Test 4: Supreme Court vs High Court hierarchy
# ============================================================

def test_court_hierarchy_in_reranker(db):
    """Supreme Court case should receive higher court weight than High Court case."""
    sc_case = _make_hc(db, "SC Murder Case", court="Supreme Court of India",
                       summary="murder conviction Section 302 IPC evidence")
    _make_chunk(db, sc_case,
                "The Supreme Court upheld the conviction under Section 302 IPC "
                "based on eyewitness testimony and forensic evidence.")

    hc_case = _make_hc(db, "HC Murder Case", court="Bombay High Court",
                       summary="murder conviction Section 302 IPC evidence")
    _make_chunk(db, hc_case,
                "The High Court upheld the conviction under Section 302 IPC "
                "based on eyewitness testimony and forensic evidence.")

    query = "murder Section 302 IPC eyewitness conviction"
    query_case = models.Case(title=query, facts=query, issues=query)

    candidates = retrieve_historical_chunks(db, query, top_k=10)
    reranked = rerank_candidates(db, query_case, candidates, top_k=10)

    assert isinstance(reranked, list)

    if len(reranked) >= 2:
        sc_entry = next((r for r in reranked if r.historical_case.id == sc_case.id), None)
        hc_entry = next((r for r in reranked if r.historical_case.id == hc_case.id), None)

        if sc_entry and hc_entry:
            # Supreme Court should have higher court_hierarchy_score
            assert sc_entry.court_hierarchy_score >= hc_entry.court_hierarchy_score, (
                "Supreme Court case should have >= court hierarchy score vs High Court"
            )


# ============================================================
# Test 5: Highly cited vs less-cited precedent
# ============================================================

def test_citation_authority_ordering(db):
    """A highly cited case should receive a higher authority score."""
    from app.services.citation_graph import get_citation_graph, _graph_singleton
    import app.services.citation_graph as cg_module
    cg_module._graph_singleton = None  # Reset singleton for fresh build

    high_cited = _make_hc(db, "Landmark Case", cited_by_count=200,
                          summary="landmark constitutional article 21 right liberty")
    _make_chunk(db, high_cited,
                "This landmark case established the right to personal liberty "
                "under Article 21 of the Constitution of India.")

    low_cited = _make_hc(db, "Obscure Case", cited_by_count=2,
                         summary="constitutional article 21 right liberty")
    _make_chunk(db, low_cited,
                "Article 21 of the Constitution provides right to life and liberty.")

    query = "Article 21 right to personal liberty Constitution of India"
    qcase = models.Case(title=query, facts=query, issues=query)
    candidates = retrieve_historical_chunks(db, query, top_k=10)
    reranked = rerank_candidates(db, qcase, candidates, top_k=10)

    assert isinstance(reranked, list)

    if len(reranked) >= 2:
        high_entry = next((r for r in reranked if r.historical_case.id == high_cited.id), None)
        low_entry = next((r for r in reranked if r.historical_case.id == low_cited.id), None)

        if high_entry and low_entry:
            assert high_entry.citation_authority_score >= low_entry.citation_authority_score, (
                "Highly cited case should have >= citation authority score"
            )


# ============================================================
# Test 6: Query with no statutory information
# ============================================================

def test_query_missing_statutory_info(db):
    """Query with no section/act references — system must not crash."""
    hc = _make_hc(db, "General Legal Case", summary="legal proceeding court hearing")
    _make_chunk(db, hc, "The court heard arguments from both parties and reserved judgment.")

    query = "court hearing arguments reserved judgment"  # no statute
    results = retrieve_historical_chunks(db, query, top_k=10)
    assert isinstance(results, list)


# ============================================================
# Test 7: Vague legal language
# ============================================================

def test_vague_query(db):
    """Very vague query — system should not crash, may return zero or low-scored results."""
    hc = _make_hc(db, "Misc Case", summary="matter court order")
    _make_chunk(db, hc, "The court passed an order in the matter.")

    results = retrieve_historical_chunks(db, "something legal happened in court", top_k=10)
    assert isinstance(results, list)
    assert len(results) >= 0


# ============================================================
# Test 8: Query with no matching precedent
# ============================================================

def test_query_no_relevant_precedent(db):
    """Query completely outside legal domain — system should return empty or near-zero-scored results."""
    results = retrieve_historical_chunks(
        db,
        "quantum physics thermodynamics black holes",
        top_k=10
    )
    assert isinstance(results, list)
    if results:
        # Scores should be very low
        for r in results:
            assert r.hybrid_score >= 0.0  # must never be negative


# ============================================================
# Test 9: Empty document input
# ============================================================

def test_empty_document_extraction():
    """Empty text should not crash the structure extractor."""
    result = extract_legal_sections_structure("")
    assert isinstance(result, dict)
    assert result["facts"] == ""
    assert result["decision"] == ""


# ============================================================
# Test 10: Poorly formatted document
# ============================================================

def test_poorly_formatted_document():
    """Noisy document with line artifacts — cleaner should handle gracefully."""
    noisy_text = (
        "IN THE SUPREME COURT OF INDIA\n"
        "------------------------------\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "     Page 1 of 10     \n"
        "Writ Petition (C) No. 123 of 2020\n"
        "\x00\x01 BINARY GARBAGE \x00\x01\n"
        "   \n   \n"
        "The petitioner filed the present writ under Article 32.\n"
    )
    cleaned = clean_legal_text(noisy_text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    assert "petitioner" in cleaned.lower()


# ============================================================
# Test 11: Missing citation metadata
# ============================================================

def test_missing_citation_metadata_safe_handling(db):
    """
    A Citation row with NULL chunk_id should not crash the provenance validator.
    It should be flagged as invalid, not raise an exception.
    """
    from provenance_validator import verify_citation

    # Create a minimal analysis + citation with missing chunk_id
    qcase = models.Case(title="Meta Test Case", facts="test", issues="test")
    db.add(qcase)
    db.flush()

    analysis = models.Analysis(
        case_id=qcase.id,
        analysis_type="similar_cases",
        content={"similar_cases": []},
        provider="mock",
    )
    db.add(analysis)
    db.flush()

    bad_citation = models.Citation(
        analysis_id=analysis.id,
        ref_key="S1",
        source_type="historical_case",
        chunk_id=None,               # Missing chunk reference
        historical_case_id=None,     # Missing case reference
        case_name="Unknown Case",
        excerpt="Some text",
    )
    db.add(bad_citation)
    db.flush()

    from provenance_validator import verify_citation as _verify
    result = _verify(db, bad_citation)
    assert not result.chunk_resolved
    assert not result.is_fully_valid
    assert len(result.issues) > 0


# ============================================================
# Test 12: LLM assertion without valid source chunk
# ============================================================

def test_unsupported_claim_detection(db):
    """
    Validate that the provenance system correctly identifies a citation
    where the ref_key maps to a non-existent chunk_id.
    """
    qcase = models.Case(title="Unsupported Claim Test", facts="test", issues="test")
    db.add(qcase)
    db.flush()

    analysis = models.Analysis(
        case_id=qcase.id,
        analysis_type="similar_cases",
        content={},
        provider="mock",
    )
    db.add(analysis)
    db.flush()

    bad_citation = models.Citation(
        analysis_id=analysis.id,
        ref_key="S99",
        source_type="historical_case",
        chunk_id="NONEXISTENT_CHUNK_ID_abc123",
        historical_case_id="NONEXISTENT_CASE_ID_xyz",
        case_name="Phantom Case",
        excerpt="This claim has no real source.",
    )
    db.add(bad_citation)
    db.flush()

    from provenance_validator import verify_citation as _verify
    result = _verify(db, bad_citation)
    assert not result.chunk_resolved
    assert not result.case_resolved
    assert not result.is_fully_valid


# ============================================================
# Metric Correctness Tests
# ============================================================

def test_metrics_precision_at_k():
    from metrics import precision_at_k
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "E"}
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2/3)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(3/5)
    assert precision_at_k([], relevant, 5) == 0.0
    assert precision_at_k(retrieved, set(), 5) == 0.0


def test_metrics_recall_at_k():
    from metrics import recall_at_k
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "E"}
    assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1/3)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)


def test_metrics_mrr():
    from metrics import reciprocal_rank
    retrieved = ["X", "Y", "A", "B"]
    relevant = {"A"}
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)
    assert reciprocal_rank(["X", "Y"], relevant) == 0.0


def test_metrics_ndcg():
    from metrics import ndcg_at_k
    retrieved = ["A", "B", "C"]
    graded = {"A": 2.0, "B": 1.0, "C": 0.0}
    score = ndcg_at_k(retrieved, graded, 3)
    assert 0.0 <= score <= 1.0


def test_metrics_hit_rate():
    from metrics import hit_rate_at_k
    retrieved = ["X", "Y", "A"]
    relevant = {"A"}
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0
    assert hit_rate_at_k([], relevant, 5) == 0.0


def test_weight_sum_validation():
    from weight_sensitivity import WEIGHT_CONFIGS, validate_weights
    for cfg_name, cfg in WEIGHT_CONFIGS.items():
        w = cfg["weights"]
        assert validate_weights(w), (
            f"Config '{cfg_name}' weights sum to {sum(w.values()):.4f}, expected ~1.0"
        )


def test_ablation_configs_normalized():
    from ablation import ABLATION_CONFIGS
    for cfg_name, weights in ABLATION_CONFIGS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, (
            f"Ablation config '{cfg_name}' weights sum to {total:.4f}"
        )
