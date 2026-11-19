import json

SYSTEM_RULES = """You are Nya(AI), a legal decision-support assistant. You are NOT a judge and must never determine guilt, innocence, sentencing or a final outcome.
Rules:
1. Use ONLY the SOURCES provided below with ref_key like [S1].
2. Never invent cases, statutes, sections, quotations, dates, or facts.
3. Every factual claim must cite the ref_key(s) that support it.
4. Clearly distinguish facts stated in sources from your own inference.
5. If something is unavailable or uncertain, explicitly say so — do not guess.
6. Absence of evidence is not evidence of absence — say "not found in the provided sources".
7. Do not reach unsupported legal conclusions; flag matters that need verification.
8. Respond ONLY in the requested JSON schema."""

def _sources_block(sources: list) -> str:
    lines = [f"[{s['ref_key']}] (case: {s.get('case_name', 'current case')}, doc: {s.get('document_filename', 'n/a')}, page: {s.get('page_number', 'n/a')})\n{s['text'][:800]}" for s in sources]
    return "\n\n".join(lines) if lines else "(no sources retrieved)"

def similar_cases_prompt(query_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Review precedent SOURCES. Select up to 5 cases sharing legal issues, land acquisition contexts, or compensation principles with the CURRENT CASE.
For each precedent provide: case_name, why_relevant (with [S1]), matching_excerpt, ref_key.

CURRENT CASE:
{query_text[:1200]}

RETRIEVED PRECEDENTS:
{_sources_block(sources)}

Respond STRICTLY with JSON: {{"similar_cases": [{{"case_name": str, "why_relevant": str, "matching_excerpt": str, "ref_key": str}}]}}"""

def legal_provisions_prompt(case_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: From SOURCES only, extract relevant statutory provisions/sections. Do not name any statute not literally in a source.
CASE: {case_text}
SOURCES:
{_sources_block(sources)}
Respond as JSON: {{"legal_provisions": [{{"provision": str, "why_relevant": str, "source_refs": [str]}}]}}"""

def prosecution_defence_prompt(case_text: str, arguments_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Produce balanced prosecution-vs-defence analysis from CASE ARGUMENTS and SOURCES.
CASE: {case_text}
CASE ARGUMENTS: {arguments_text}
SOURCES:
{_sources_block(sources)}
Respond as JSON: {{"prosecution": {{"arguments": [str], "supporting_precedents": [str], "gaps": [str]}}, "defence": {{"arguments": [str], "supporting_precedents": [str], "gaps": [str]}}}}"""

def evidence_gaps_prompt(evidence_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Separate evidence actually present from potential evidence to verify.
EVIDENCE TEXT: {evidence_text}
SOURCES:
{_sources_block(sources)}
Respond as JSON: {{"evidence_present": [str], "potential_evidence_to_verify": [str]}}"""

def contradictions_prompt(sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Identify genuine contradictions across SOURCES citing ref_keys. If none found, state explicitly.
SOURCES:
{_sources_block(sources)}
Respond as JSON: {{"contradictions": [{{"type": str, "description": str, "source_refs": [str]}}]}}"""

def comparison_prompt(current_case: dict, historical_case: dict, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Compare CURRENT CASE with HISTORICAL CASE. Ground claims in text.
CURRENT CASE: {json.dumps(current_case, indent=2)}
HISTORICAL CASE: {json.dumps(historical_case, indent=2)}
SOURCES:
{_sources_block(sources)}
Respond as JSON: {{"similarities": [str], "differences": {{"unique_to_current_case": [str], "unique_to_historical_case": [str]}}, "legal_significance": str, "applicability_considerations": [str]}}"""

def brief_prompt(case: dict, prior_analyses: dict) -> str:
    return f"""{SYSTEM_RULES}

TASK: Assemble pre-hearing brief strictly from CASE data and PRIOR ANALYSES.
CASE: {json.dumps(case, indent=2)}
PRIOR ANALYSES: {json.dumps(prior_analyses, indent=2)}
Respond as JSON: {{"case_overview": str, "key_facts": str, "legal_issues": str, "relevant_laws": [], "relevant_precedents": [], "prosecution_arguments": {{}}, "defence_arguments": {{}}, "evidence_summary": [], "evidence_gaps": [], "contradictions": [], "questions_to_examine": [str]}}"""
