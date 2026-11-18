"""Orchestrates the RAG pipeline for every analysis type:

  gather case + retrieve sources -> build citation-tagged context package
  -> prompt -> LLMProvider -> structured result -> persist Analysis + Citation
     rows -> return result (with citations attached) to the API layer.

This is the one place that ties retrieval, prompts, and the LLM provider
together, so each analysis type is a thin function here.
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from .. import models
from . import prompts
from .llm_provider import get_llm_provider
from .retrieval import retrieve_historical_chunks, retrieve_case_document_chunks, aggregate_by_historical_case


def _case_query_text(case: models.Case) -> str:
    return "\n".join(filter(None, [
        f"Case type: {case.case_type or ''}",
        f"Facts: {case.facts or ''}",
        f"Charges/Issues: {case.charges_issues or ''}",
        f"Arguments: {case.arguments_text or ''}",
    ]))


def _make_source_entry(ref_key: str, chunk: models.Chunk, case_name: Optional[str],
                        score: Optional[float] = None) -> Dict[str, Any]:
    return {
        "ref_key": ref_key,
        "chunk_id": chunk.id,
        "historical_case_id": chunk.historical_case_id,
        "case_name": case_name,
        "document_filename": chunk.document.filename if chunk.document else None,
        "page_number": chunk.page_number,
        "text": chunk.text,
        "hybrid_score": score,
    }


def _persist_citations(db: Session, analysis: models.Analysis, sources: List[Dict[str, Any]]):
    for s in sources:
        db.add(models.Citation(
            analysis_id=analysis.id,
            ref_key=s["ref_key"],
            source_type="historical_case" if s.get("historical_case_id") else "current_document",
            chunk_id=s.get("chunk_id"),
            historical_case_id=s.get("historical_case_id"),
            case_name=s.get("case_name"),
            document_filename=s.get("document_filename"),
            page_number=s.get("page_number"),
            excerpt=(s.get("text") or "")[:500],
            similarity_score=s.get("hybrid_score"),
        ))
    db.commit()


def _save_analysis(db: Session, case_id: str, analysis_type: str, content: dict,
                    provider_name: str, sources: List[Dict[str, Any]]) -> models.Analysis:
    analysis = models.Analysis(case_id=case_id, analysis_type=analysis_type,
                                content=content, provider=provider_name)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    _persist_citations(db, analysis, sources)
    return analysis


def _retrieved_to_sources(db: Session, retrieved) -> List[Dict[str, Any]]:
    sources = []
    hc_ids = list({r.chunk.historical_case_id for r in retrieved if r.chunk.historical_case_id})
    hc_cache: Dict[str, models.HistoricalCase] = {}
    if hc_ids:
        for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(hc_ids)).all():
            hc_cache[hc.id] = hc
    for i, r in enumerate(retrieved, start=1):
        hc = hc_cache.get(r.chunk.historical_case_id)
        sources.append(_make_source_entry(f"S{i}", r.chunk, hc.case_name if hc else None, r.hybrid_score))
    return sources


def run_similar_cases(db: Session, case: models.Case) -> Dict[str, Any]:
    from .reranker import rerank_candidates
    query_text = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, query_text, top_k=20, per_case_cap=3)
    
    # Run multi-signal reranking (semantic + lexical + statutes + issue alignment + citation authority + court hierarchy)
    reranked = rerank_candidates(db, case, retrieved, top_k=8)
    
    # Use reranked chunks if available, else fallback to raw retrieved
    selected_chunks = [r.chunk for r in reranked] if reranked else [r.chunk for r in retrieved]
    reranked_dict = {r.chunk.id: r for r in reranked}

    groups = aggregate_by_historical_case(retrieved)
    sources = _retrieved_to_sources(db, retrieved)
    ref_by_chunk = {s["chunk_id"]: s["ref_key"] for s in sources}

    # Batch query historical cases for all groups
    all_hc_ids = [hc_id for hc_id in groups.keys() if hc_id]
    hcs_by_id = {}
    if all_hc_ids:
        for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(all_hc_ids)).all():
            hcs_by_id[hc.id] = hc

    retrieved_groups = []
    for hc_id, items in groups.items():
        hc = hcs_by_id.get(hc_id)
        if not hc:
            continue
        retrieved_groups.append({
            "historical_case": {
                "id": hc.id, "case_name": hc.case_name, "court": hc.court,
                "year": hc.year, "citation": hc.citation, "is_synthetic": hc.is_synthetic,
            },
            "chunks": [
                {"text": it.chunk.text, "hybrid_score": it.hybrid_score,
                 "ref_key": ref_by_chunk.get(it.chunk.id, "S1")}
                for it in items
            ],
        })

    provider = get_llm_provider()
    prompt = prompts.similar_cases_prompt(query_text, sources)
    result = provider.generate_structured(prompt, {
        "query_text": query_text, "retrieved_groups": retrieved_groups, "sources": sources,
    }, analysis_type="similar_cases")

    # Augment similar cases with reranking score breakdown and explanation
    if "similar_cases" in result and isinstance(result["similar_cases"], list):
        for sc in result["similar_cases"]:
            # Find matching reranked entry
            match = next((r for r in reranked if r.historical_case.case_name == sc.get("case_name")), None)
            if match:
                sc["rerank_total_score"] = round(match.total_score, 3)
                sc["score_breakdown"] = match.to_dict()["score_breakdown"]
                if not sc.get("why_relevant") or "closest available" in sc.get("why_relevant", ""):
                    sc["why_relevant"] = match.explanation

    analysis = _save_analysis(db, case.id, "similar_cases", result,
                               provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_legal_provisions(db: Session, case: models.Case) -> Dict[str, Any]:
    query_text = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, query_text, top_k=10)
    own_chunks = retrieve_case_document_chunks(db, case.id, query_text, top_k=6)
    sources = _retrieved_to_sources(db, retrieved)
    for j, c in enumerate(own_chunks, start=len(sources) + 1):
        sources.append(_make_source_entry(f"S{j}", c, "current case document"))

    provider = get_llm_provider()
    prompt = prompts.legal_provisions_prompt(query_text, sources)
    result = provider.generate_structured(prompt, {"sources": sources}, analysis_type="legal_provisions")
    analysis = _save_analysis(db, case.id, "legal_provisions", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_prosecution_defence(db: Session, case: models.Case) -> Dict[str, Any]:
    query_text = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, query_text, top_k=8)
    sources = _retrieved_to_sources(db, retrieved)

    # Prefer structured arguments dict if present, otherwise fallback to arguments_text
    args_content = case.arguments if case.arguments else (case.arguments_text or "")
    if isinstance(args_content, dict):
        args_str = "\n".join([f"Petitioner: {p}" for p in args_content.get("petitioner", [])] +
                             [f"Respondent: {r}" for r in args_content.get("respondent", [])])
    else:
        args_str = str(args_content)

    provider = get_llm_provider()
    prompt = prompts.prosecution_defence_prompt(query_text, args_str, sources)
    result = provider.generate_structured(prompt, {
        "case_arguments": args_content, "sources": sources,
    }, analysis_type="prosecution_defence")
    analysis = _save_analysis(db, case.id, "prosecution_defence", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_evidence_gaps(db: Session, case: models.Case) -> Dict[str, Any]:
    own_chunks = retrieve_case_document_chunks(db, case.id, top_k=20)
    sources = [_make_source_entry(f"S{i}", c, "current case document")
               for i, c in enumerate(own_chunks, start=1)]

    provider = get_llm_provider()
    prompt = prompts.evidence_gaps_prompt(case.evidence_text or "", sources)
    result = provider.generate_structured(prompt, {
        "evidence_text": case.evidence_text, "sources": sources,
    }, analysis_type="evidence_gaps")
    analysis = _save_analysis(db, case.id, "evidence_gaps", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_contradictions(db: Session, case: models.Case) -> Dict[str, Any]:
    own_chunks = retrieve_case_document_chunks(db, case.id, top_k=30)
    sources = [_make_source_entry(f"S{i}", c, "current case document")
               for i, c in enumerate(own_chunks, start=1)]

    provider = get_llm_provider()
    prompt = prompts.contradictions_prompt(sources)
    result = provider.generate_structured(prompt, {"sources": sources}, analysis_type="contradictions")
    analysis = _save_analysis(db, case.id, "contradictions", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_comparison(db: Session, case: models.Case, historical_case_id: str) -> Dict[str, Any]:
    hc = db.get(models.HistoricalCase, historical_case_id)
    if hc is None:
        raise ValueError("Historical case not found")

    query_text = _case_query_text(case)
    hc_chunks = db.query(models.Chunk).filter(models.Chunk.historical_case_id == hc.id).limit(6).all()
    sources = [_make_source_entry(f"S{i}", c, hc.case_name) for i, c in enumerate(hc_chunks, start=1)]

    current_payload = {"title": case.title, "facts": case.facts, "issues": case.charges_issues,
                        "arguments": case.arguments_text, "evidence": case.evidence_text}
    historical_payload = {"case_name": hc.case_name, "court": hc.court, "year": hc.year,
                           "summary": hc.summary, "outcome": hc.outcome}

    provider = get_llm_provider()
    prompt = prompts.comparison_prompt(current_payload, historical_payload, sources)
    result = provider.generate_structured(prompt, {
        "current_case": current_payload, "historical_case": historical_payload, "sources": sources,
    }, analysis_type="comparison")
    result["historical_case"] = {"id": hc.id, "case_name": hc.case_name, "court": hc.court,
                                  "year": hc.year, "summary": hc.summary, "outcome": hc.outcome,
                                  "is_synthetic": hc.is_synthetic}
    result["current_case"] = current_payload
    analysis = _save_analysis(db, case.id, "comparison", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}


def run_brief(db: Session, case: models.Case) -> Dict[str, Any]:
    prior = {}
    for a_type in ["similar_cases", "legal_provisions", "prosecution_defence", "evidence_gaps", "contradictions"]:
        latest = (db.query(models.Analysis)
                  .filter(models.Analysis.case_id == case.id, models.Analysis.analysis_type == a_type)
                  .order_by(models.Analysis.created_at.desc()).first())
        if latest:
            prior[a_type] = latest.content

    case_payload = {"title": case.title, "case_type": case.case_type, "facts": case.facts,
                     "charges_issues": case.charges_issues}

    provider = get_llm_provider()
    prompt = prompts.brief_prompt(case_payload, prior)
    ctx = {
        "case": case_payload,
        "legal_provisions": prior.get("legal_provisions", {}).get("legal_provisions", []),
        "similar_cases": prior.get("similar_cases", {}).get("similar_cases", []),
        "prosecution": prior.get("prosecution_defence", {}).get("prosecution", {}),
        "defence": prior.get("prosecution_defence", {}).get("defence", {}),
        "evidence_present": prior.get("evidence_gaps", {}).get("evidence_present", []),
        "evidence_gaps": prior.get("evidence_gaps", {}).get("potential_evidence_to_verify", []),
        "contradictions": prior.get("contradictions", {}).get("contradictions", []),
    }
    result = provider.generate_structured(prompt, ctx, analysis_type="brief")
    if not prior:
        result["warning"] = "No prior analyses found. Run similar cases, legal provisions, prosecution/defence, evidence, and contradictions first for a fully grounded brief."
    analysis = _save_analysis(db, case.id, "brief", result, provider.__class__.__name__, [])
    return {"analysis_id": analysis.id, **result, "citations": []}


def _citations_payload(analysis: models.Analysis) -> List[Dict[str, Any]]:
    return [{
        "ref_key": c.ref_key, "source_type": c.source_type, "case_name": c.case_name,
        "document_filename": c.document_filename, "page_number": c.page_number,
        "excerpt": c.excerpt, "similarity_score": c.similarity_score,
        "chunk_id": c.chunk_id, "historical_case_id": c.historical_case_id,
    } for c in analysis.citations]
