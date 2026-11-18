"""Structured prompt templates.

Every prompt embeds the SYSTEM_RULES so that whichever LLMProvider is active
(mock or real), the same grounding constraints are documented and, for the
real provider, actually sent to the model.
"""
import json

SYSTEM_RULES = """You are Nya(AI), a legal decision-support assistant. You are NOT a judge
and must never determine guilt, innocence, sentencing or a final legal outcome.

Rules you must follow strictly:
1. Use ONLY the SOURCES provided below. Every source has a ref_key like [S1].
2. Never invent a case, statute, section, quotation, date, or fact that is not
   present in the SOURCES or the CASE INPUT.
3. Every factual claim you make must cite the ref_key(s) that support it.
4. Clearly distinguish facts stated in sources from your own inference.
5. If something is unavailable or uncertain, explicitly say so — do not guess.
6. Absence of evidence is not evidence of absence — say "not found in the
   provided sources", not "did not happen".
7. Do not reach unsupported legal conclusions; flag matters that need
   verification by a licensed legal professional.
8. Respond ONLY in the requested JSON schema."""


def _sources_block(sources: list) -> str:
    lines = []
    for s in sources:
        lines.append(
            f"[{s['ref_key']}] (case: {s.get('case_name', 'current case')}, "
            f"doc: {s.get('document_filename', 'n/a')}, page: {s.get('page_number', 'n/a')})\n"
            f"{s['text'][:800]}"
        )
    return "\n\n".join(lines) if lines else "(no sources retrieved)"


def similar_cases_prompt(query_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Review the retrieved precedent SOURCES below. Select up to 5 cases that share relevant legal issues, land acquisition contexts, compensation principles, or statutory provisions with the CURRENT CASE.
For each selected precedent, provide:
1. case_name: The name of the precedent case.
2. why_relevant: Explanation of why this precedent is relevant to the legal or factual issues of the current case, citing its [ref_key].
3. matching_excerpt: A concise excerpt from the source text demonstrating the similarity.
4. ref_key: The reference key (e.g. S1, S2).

CURRENT CASE:
{query_text[:1200]}

RETRIEVED PRECEDENTS:
{_sources_block(sources)}

Respond STRICTLY with valid JSON in this schema:
{{"similar_cases": [{{"case_name": "Case Title", "why_relevant": "Explanation with [S1]", "matching_excerpt": "Relevant quote", "ref_key": "S1"}}]}}"""


def legal_provisions_prompt(case_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: From the SOURCES only, extract statutory provisions, sections, or legal
principles that appear to be relevant to this case. Do not name any statute
that does not literally appear in a source.

CASE:
{case_text}

SOURCES:
{_sources_block(sources)}

Respond as JSON: {{"legal_provisions": [{{"provision": str, "why_relevant": str,
"source_refs": [str]}}]}}"""


def prosecution_defence_prompt(case_text: str, arguments_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Produce a balanced prosecution-vs-defence analysis. Do not favor either
side. Base arguments only on the CASE ARGUMENTS text and SOURCES.

CASE:
{case_text}

CASE ARGUMENTS (as entered by the user):
{arguments_text}

SOURCES:
{_sources_block(sources)}

Respond as JSON: {{"prosecution": {{"arguments": [str], "supporting_precedents": [str],
"gaps": [str]}}, "defence": {{"arguments": [str], "supporting_precedents": [str],
"gaps": [str]}}}}"""


def evidence_gaps_prompt(evidence_text: str, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Separate evidence that is actually present (from EVIDENCE TEXT/SOURCES)
from evidence that is merely potentially relevant and should be verified.
Never claim potential evidence already exists.

EVIDENCE TEXT:
{evidence_text}

SOURCES:
{_sources_block(sources)}

Respond as JSON: {{"evidence_present": [str], "potential_evidence_to_verify": [str]}}"""


def contradictions_prompt(sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Compare information across the SOURCES and identify genuine contradictions
(conflicting statements, inconsistent timelines/dates, conflicting evidence,
inconsistent arguments). For every contradiction, cite the ref_keys involved.
If none are found, say so explicitly rather than inventing one.

SOURCES:
{_sources_block(sources)}

Respond as JSON: {{"contradictions": [{{"type": str, "description": str,
"source_refs": [str]}}]}}"""


def comparison_prompt(current_case: dict, historical_case: dict, sources: list) -> str:
    return f"""{SYSTEM_RULES}

TASK: Compare the CURRENT CASE with the HISTORICAL CASE. Identify similarities,
differences, legal significance, and applicability considerations. Ground every
claim in the provided text.

CURRENT CASE:
{json.dumps(current_case, indent=2)}

HISTORICAL CASE:
{json.dumps(historical_case, indent=2)}

SOURCES:
{_sources_block(sources)}

Respond as JSON: {{"similarities": [str], "differences": {{"unique_to_current_case": [str],
"unique_to_historical_case": [str]}}, "legal_significance": str,
"applicability_considerations": [str]}}"""


def brief_prompt(case: dict, prior_analyses: dict) -> str:
    return f"""{SYSTEM_RULES}

TASK: Assemble a structured pre-hearing brief strictly from the CASE data and
PRIOR ANALYSES results below (each of which was itself source-grounded). Do
not add any new facts, cases, or laws not already present in these inputs.

CASE:
{json.dumps(case, indent=2)}

PRIOR ANALYSES:
{json.dumps(prior_analyses, indent=2)}

Respond as JSON: {{"case_overview": str, "key_facts": str, "legal_issues": str,
"relevant_laws": [], "relevant_precedents": [], "prosecution_arguments": {{}},
"defence_arguments": {{}}, "evidence_summary": [], "evidence_gaps": [],
"contradictions": [], "questions_to_examine": [str]}}"""
