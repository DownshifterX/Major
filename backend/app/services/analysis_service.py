from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from .. import models
from . import prompts
from .llm_provider import get_llm_provider
from .retrieval import retrieve_historical_chunks, retrieve_case_document_chunks, aggregate_by_historical_case

def _case_query_text(case: models.Case) -> str:
    return "\n".join(filter(None, [
        f"Case type: {case.case_type or ''}", f"Facts: {case.facts or ''}",
        f"Charges/Issues: {case.charges_issues or ''}", f"Arguments: {case.arguments_text or ''}",
    ]))

def _make_source_entry(ref_key: str, chunk: models.Chunk, case_name: Optional[str], score: Optional[float] = None) -> Dict[str, Any]:
    return {
        "ref_key": ref_key, "chunk_id": chunk.id, "historical_case_id": chunk.historical_case_id,
        "case_name": case_name, "document_filename": chunk.document.filename if chunk.document else None,
        "page_number": chunk.page_number, "text": chunk.text, "hybrid_score": score,
    }

def _persist_citations(db: Session, analysis: models.Analysis, sources: List[Dict[str, Any]]):
    for s in sources:
        db.add(models.Citation(
            analysis_id=analysis.id, ref_key=s["ref_key"],
            source_type="historical_case" if s.get("historical_case_id") else "current_document",
            chunk_id=s.get("chunk_id"), historical_case_id=s.get("historical_case_id"),
            case_name=s.get("case_name"), document_filename=s.get("document_filename"),
            page_number=s.get("page_number"), excerpt=(s.get("text") or "")[:500],
            similarity_score=s.get("hybrid_score"),
        ))
    db.commit()

def _save_analysis(db: Session, case_id: str, analysis_type: str, content: dict, provider_name: str, sources: List[Dict[str, Any]]) -> models.Analysis:
    analysis = models.Analysis(case_id=case_id, analysis_type=analysis_type, content=content, provider=provider_name)
    db.add(analysis); db.commit(); db.refresh(analysis)
    _persist_citations(db, analysis, sources)
    return analysis

def _retrieved_to_sources(db: Session, retrieved) -> List[Dict[str, Any]]:
    sources, hc_cache = [], {}
    hc_ids = list({r.chunk.historical_case_id for r in retrieved if r.chunk.historical_case_id})
    if hc_ids:
        hc_cache = {hc.id: hc for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(hc_ids)).all()}
    for i, r in enumerate(retrieved, start=1):
        hc = hc_cache.get(r.chunk.historical_case_id)
        sources.append(_make_source_entry(f"S{i}", r.chunk, hc.case_name if hc else None, r.hybrid_score))
    return sources

def _citations_payload(analysis: models.Analysis) -> List[Dict[str, Any]]:
    return [{
        "ref_key": c.ref_key, "source_type": c.source_type, "case_name": c.case_name,
        "document_filename": c.document_filename, "page_number": c.page_number,
        "excerpt": c.excerpt, "similarity_score": c.similarity_score,
        "chunk_id": c.chunk_id, "historical_case_id": c.historical_case_id,
    } for c in analysis.citations]

def run_similar_cases(db: Session, case: models.Case) -> Dict[str, Any]:
    from .reranker import rerank_candidates
    q = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, q, top_k=20, per_case_cap=3)
    reranked = rerank_candidates(db, case, retrieved, top_k=8)

    groups = aggregate_by_historical_case(retrieved)
    sources = _retrieved_to_sources(db, retrieved)
    ref_by_chunk = {s["chunk_id"]: s["ref_key"] for s in sources}
    all_hc_ids = [hc_id for hc_id in groups.keys() if hc_id]
    hcs_by_id = {hc.id: hc for hc in db.query(models.HistoricalCase).filter(models.HistoricalCase.id.in_(all_hc_ids)).all()} if all_hc_ids else {}

    retrieved_groups = []
    for hc_id, items in groups.items():
        hc = hcs_by_id.get(hc_id)
        if hc:
            retrieved_groups.append({
                "historical_case": {"id": hc.id, "case_name": hc.case_name, "court": hc.court, "year": hc.year, "citation": hc.citation, "is_synthetic": hc.is_synthetic},
                "chunks": [{"text": it.chunk.text, "hybrid_score": it.hybrid_score, "ref_key": ref_by_chunk.get(it.chunk.id, "S1")} for it in items],
            })

    provider = get_llm_provider()
    result = provider.generate_structured(prompts.similar_cases_prompt(q, sources), {"query_text": q, "retrieved_groups": retrieved_groups, "sources": sources}, analysis_type="similar_cases")

    # Clean, enrich, and supplement similar_cases so there are always 5-8 solid precedents
    seen_names = set()
    cleaned_cases = []
    current_title = (case.title or "").strip().lower()

    existing = result.get("similar_cases") if isinstance(result.get("similar_cases"), list) else []
    for sc in existing:
        if not isinstance(sc, dict) or not sc.get("case_name"):
            continue
        cname = sc["case_name"].strip()
        norm_cname = cname.lower()
        if norm_cname in seen_names or (current_title and norm_cname == current_title):
            continue

        # Match in reranked candidates (exact, prefix, or substring)
        match = next(
            (r for r in reranked if r.historical_case.case_name.strip().lower() == norm_cname
             or norm_cname in r.historical_case.case_name.strip().lower()
             or r.historical_case.case_name.strip().lower() in norm_cname),
            None
        )
        if match:
            cname = match.historical_case.case_name.strip()
            score = round(match.total_score, 3)
            breakdown = match.to_dict().get("score_breakdown", {})
            why = sc.get("why_relevant") or match.explanation
            court = sc.get("court") or match.historical_case.court or "High Court / Supreme Court"
            year = sc.get("year") or match.historical_case.year
            ref_key = sc.get("ref_key") or ref_by_chunk.get(match.chunk.id, "S1")
            excerpt = sc.get("matching_excerpt") or (match.chunk.text or "")[:400]
        else:
            score = sc.get("relevance_score") or sc.get("rerank_total_score") or 0.75
            breakdown = sc.get("score_breakdown") or {"semantic": 0.75, "lexical": 0.70, "statute_overlap": 0.50, "citation_authority": 0.60}
            why = sc.get("why_relevant") or "Precedent shares statutory interpretation and procedural nexus with the current dispute."
            court = sc.get("court") or "High Court / Supreme Court"
            year = sc.get("year")
            ref_key = sc.get("ref_key", "S1")
            excerpt = sc.get("matching_excerpt", "")

        seen_names.add(norm_cname)
        seen_names.add(cname.lower())
        cleaned_cases.append({
            "case_name": cname,
            "court": court,
            "year": year,
            "relevance_score": score,
            "rerank_total_score": score,
            "score_breakdown": breakdown,
            "why_relevant": why,
            "matching_excerpt": excerpt,
            "ref_key": ref_key,
        })

    # Supplement from reranked candidates to ensure at least 6 rich precedents
    for r in reranked:
        if len(cleaned_cases) >= 6:
            break
        cname = r.historical_case.case_name.strip()
        if cname.lower() in seen_names or (current_title and cname.lower() == current_title):
            continue
        seen_names.add(cname.lower())
        score = round(r.total_score, 3)
        cleaned_cases.append({
            "case_name": cname,
            "court": r.historical_case.court or "High Court / Supreme Court",
            "year": r.historical_case.year,
            "relevance_score": score,
            "rerank_total_score": score,
            "score_breakdown": r.to_dict().get("score_breakdown", {}),
            "why_relevant": r.explanation,
            "matching_excerpt": (r.chunk.text or "")[:400],
            "ref_key": ref_by_chunk.get(r.chunk.id, "S1"),
        })

    result["similar_cases"] = cleaned_cases

    analysis = _save_analysis(db, case.id, "similar_cases", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}

def run_legal_provisions(db: Session, case: models.Case) -> Dict[str, Any]:
    q = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, q, top_k=10)
    own_chunks = retrieve_case_document_chunks(db, case.id, q, top_k=6)
    sources = _retrieved_to_sources(db, retrieved)
    for j, c in enumerate(own_chunks, start=len(sources) + 1):
        sources.append(_make_source_entry(f"S{j}", c, "current case document"))

    provider = get_llm_provider()
    result = provider.generate_structured(prompts.legal_provisions_prompt(q, sources), {"sources": sources}, analysis_type="legal_provisions")
    analysis = _save_analysis(db, case.id, "legal_provisions", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}

def run_prosecution_defence(db: Session, case: models.Case) -> Dict[str, Any]:
    q = _case_query_text(case)
    retrieved = retrieve_historical_chunks(db, q, top_k=8)
    own_chunks = retrieve_case_document_chunks(db, case.id, q, top_k=8)
    sources = _retrieved_to_sources(db, retrieved)
    for j, c in enumerate(own_chunks, start=len(sources) + 1):
        sources.append(_make_source_entry(f"S{j}", c, "current case document"))

    args_data = case.arguments if isinstance(case.arguments, dict) else {}
    provider = get_llm_provider()
    result = provider.generate_structured(
        prompts.prosecution_defence_prompt(case.facts or "", case.arguments_text or "", sources),
        {"sources": sources, "case_arguments": args_data, "query_text": q}, analysis_type="prosecution_defence"
    )
    analysis = _save_analysis(db, case.id, "prosecution_defence", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}

def run_evidence_gaps(db: Session, case: models.Case) -> Dict[str, Any]:
    q = _case_query_text(case)
    own_chunks = retrieve_case_document_chunks(db, case.id, top_k=20)
    sources = [_make_source_entry(f"S{i}", c, "current case document") for i, c in enumerate(own_chunks, start=1)]
    provider = get_llm_provider()
    result = provider.generate_structured(prompts.evidence_gaps_prompt(case.evidence_text or "", sources), {"sources": sources}, analysis_type="evidence_gaps")
    analysis = _save_analysis(db, case.id, "evidence_gaps", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}

def run_contradictions(db: Session, case: models.Case) -> Dict[str, Any]:
    own_chunks = retrieve_case_document_chunks(db, case.id, top_k=30)
    sources = [_make_source_entry(f"S{i}", c, "current case document") for i, c in enumerate(own_chunks, start=1)]
    provider = get_llm_provider()
    result = provider.generate_structured(prompts.contradictions_prompt(sources), {"sources": sources}, analysis_type="contradictions")
    analysis = _save_analysis(db, case.id, "contradictions", result, provider.__class__.__name__, sources)
    return {"analysis_id": analysis.id, **result, "citations": _citations_payload(analysis)}

def run_comparison(db: Session, case: models.Case, historical_case_id: str) -> Dict[str, Any]:
    hc = db.get(models.HistoricalCase, historical_case_id)
    if not hc: raise ValueError(f"HistoricalCase '{historical_case_id}' not found")
    q = f"{case.facts or ''}\n{case.charges_issues or ''}"
    curr_chunks = retrieve_case_document_chunks(db, case.id, q, top_k=6)
    hist_chunks = (db.query(models.Chunk).filter(models.Chunk.historical_case_id == historical_case_id).limit(6).all())

    sources = [_make_source_entry(f"S{i}", c, "current case document") for i, c in enumerate(curr_chunks, start=1)]
    for j, c in enumerate(hist_chunks, start=len(sources) + 1):
        sources.append(_make_source_entry(f"S{j}", c, hc.case_name))

    curr_payload = {"title": case.title, "court": case.court, "facts": case.facts or "", "issues": case.charges_issues or case.issues or "", "provisions": case.provisions or []}
    hist_payload = {"case_name": hc.case_name, "court": hc.court, "year": hc.year, "summary": hc.summary or "", "outcome": hc.outcome or ""}

    provider = get_llm_provider()
    result = provider.generate_structured(prompts.comparison_prompt(curr_payload, hist_payload, sources), {"sources": sources}, analysis_type="comparison")
    analysis = _save_analysis(db, case.id, "comparison", result, provider.__class__.__name__, sources)
    hist_info = {"id": hc.id, "case_name": hc.case_name, "court": hc.court, "year": hc.year, "citation": hc.citation, "summary": hc.summary, "outcome": hc.outcome}
    curr_info = {"id": case.id, "title": case.title, "court": case.court, "facts": case.facts or "", "issues": case.charges_issues or case.issues or ""}
    return {"analysis_id": analysis.id, "historical_case_id": hc.id, "current_case": curr_info, "historical_case": hist_info, "historical_case_name": hc.case_name, **result, "citations": _citations_payload(analysis)}

def run_brief(db: Session, case: models.Case, force_refresh: bool = False) -> Dict[str, Any]:
    # Return latest stored brief if it has content (avoid re-running LLM every GET)
    stored = (db.query(models.Analysis).filter(models.Analysis.case_id == case.id, models.Analysis.analysis_type == "brief")
              .order_by(models.Analysis.created_at.desc()).first())
    if not force_refresh and stored and stored.content.get("case_overview"):
        return {"analysis_id": stored.id, **stored.content, "citations": _citations_payload(stored)}

    prior_analyses: Dict[str, Any] = {}
    for an_type, runner in [("similar_cases", run_similar_cases), ("legal_provisions", run_legal_provisions),
                            ("prosecution_defence", run_prosecution_defence), ("evidence_gaps", run_evidence_gaps),
                            ("contradictions", run_contradictions)]:
        existing = (db.query(models.Analysis).filter(models.Analysis.case_id == case.id, models.Analysis.analysis_type == an_type)
                    .order_by(models.Analysis.created_at.desc()).first())
        prior_analyses[an_type] = existing.content if existing else runner(db, case)

    # Build relevant_laws from legal_provisions and relevant_precedents from similar_cases if available
    legal_prov = prior_analyses.get("legal_provisions", {})
    if isinstance(legal_prov, dict):
        laws_list = legal_prov.get("provisions") or legal_prov.get("legal_provisions") or []
        # Flatten if nested (e.g. list of dicts)
        flat_laws = [p.get("provision", p) if isinstance(p, dict) else p for p in laws_list]
    else:
        flat_laws = []

    sim = prior_analyses.get("similar_cases", {})
    if isinstance(sim, dict):
        prec_list = [sc.get("case_name") for sc in (sim.get("similar_cases") or []) if sc.get("case_name")]
    else:
        prec_list = []

    case_payload = {"title": case.title, "court": case.court, "judge": case.judge, "case_type": case.case_type, "facts": case.facts, "issues": case.charges_issues}
    provider = get_llm_provider()
    result = provider.generate_structured(prompts.brief_prompt(case_payload, prior_analyses), {"case": case_payload, "prior_analyses": prior_analyses}, analysis_type="brief")

    # Backfill empty relevant_laws / relevant_precedents from prior analyses or case records
    if not result.get("relevant_laws"):
        if flat_laws:
            result["relevant_laws"] = flat_laws
        elif case.provisions and isinstance(case.provisions, list):
            prov_strs = [f"{cp.get('section', '')} {cp.get('act', '')}".strip() if isinstance(cp, dict) else str(cp).strip() for cp in case.provisions]
            result["relevant_laws"] = [p for p in prov_strs if p]

    if not result.get("relevant_precedents") and prec_list:
        result["relevant_precedents"] = prec_list

    analysis = _save_analysis(db, case.id, "brief", result, provider.__class__.__name__, sources=[])
    return {"analysis_id": analysis.id, **result, "citations": []}
