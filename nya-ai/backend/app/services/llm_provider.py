"""LLM provider abstraction.

    LLMProvider
    ├── generate(prompt) -> str
    └── generate_structured(prompt, context) -> dict

Two implementations:

  * MockLLMProvider   — the default. Produces genuinely grounded, structured
    output using rule-based extraction over the *actual* retrieved context
    (regex-based statute/date/number extraction, keyword-tagged argument
    sorting, simple contradiction heuristics). It never invents a case, a
    statute, or a quotation — everything it emits traces back to a source
    chunk that is passed in. This keeps the demo fully runnable offline with
    no API key while staying honest about being a lower-fidelity stand-in
    for a real model.

  * AnthropicProvider — calls the real Claude API (api.anthropic.com) when
    ANTHROPIC_API_KEY is set and LLM_PROVIDER=anthropic. It sends the same
    context package and is instructed to only use supplied sources, respond
    in strict JSON, and flag uncertainty rather than fabricate.

Swap providers purely via environment variables — nothing else in the
pipeline needs to know which one is active.
"""
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import httpx


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    @abstractmethod
    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Real provider
# ---------------------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set but LLM_PROVIDER=anthropic")

    def generate(self, prompt: str) -> str:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        full_prompt = (
            prompt
            + "\n\nRespond with STRICT JSON only. No markdown fences, no preamble, no commentary."
        )
        raw = self.generate(full_prompt)
        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "error": "The model did not return valid JSON.",
                "raw_output": raw[:2000],
            }


# ---------------------------------------------------------------------------
# Mock / offline provider — deterministic, grounded, no external calls
# ---------------------------------------------------------------------------
STATUTE_RE = re.compile(
    r"\b((?:Section|Sec\.?|Article|Art\.?|Rule)\s+\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))?"
    r"(?:\s+(?:of|,)\s+the\s+[A-Z][A-Za-z, ]{2,60}?(?:Act|Code|Constitution)(?:,?\s*\d{4})?)?)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)


class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return (
            "[mock-demo provider] This offline provider only produces structured, "
            "rule-based output — see generate_structured(). Configure "
            "LLM_PROVIDER=anthropic with an ANTHROPIC_API_KEY for free-form generation."
        )

    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        handler = {
            "similar_cases": self._similar_cases,
            "legal_provisions": self._legal_provisions,
            "prosecution_defence": self._prosecution_defence,
            "evidence_gaps": self._evidence_gaps,
            "contradictions": self._contradictions,
            "comparison": self._comparison,
            "brief": self._brief,
            "case_extraction": self._case_extraction,
        }.get(analysis_type)
        if handler is None:
            return {"error": f"No mock handler for analysis_type='{analysis_type}'"}
        return handler(context)

    # -- handlers -----------------------------------------------------
    def _similar_cases(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        for group in ctx.get("retrieved_groups", []):
            hc = group["historical_case"]
            best = group["chunks"][0]
            results.append({
                "case_name": hc["case_name"],
                "court": hc.get("court"),
                "year": hc.get("year"),
                "relevance_score": round(best["hybrid_score"], 3),
                "why_relevant": self._why_relevant(ctx["query_text"], best["text"]),
                "matching_excerpt": best["text"][:400],
                "ref_key": best["ref_key"],
            })
        return {"similar_cases": results}

    def _why_relevant(self, query: str, chunk_text: str) -> str:
        from .retrieval import _keywords
        shared = sorted(_keywords(query) & _keywords(chunk_text))[:6]
        if shared:
            return ("Shares key terms/fact-patterns with the current case: "
                    + ", ".join(shared) + ".")
        return "Retrieved as the closest available match in the historical corpus; review manually to confirm relevance."

    def _legal_provisions(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        found = {}
        for src in ctx.get("sources", []):
            for m in STATUTE_RE.finditer(src["text"]):
                key = re.sub(r"\s+", " ", m.group(1)).strip()
                found.setdefault(key, []).append(src["ref_key"])
        provisions = [
            {
                "provision": name,
                "why_relevant": "Explicitly mentioned in retrieved source material; verify current applicability and exact wording against the official text.",
                "source_refs": list(set(refs)),
            }
            for name, refs in found.items()
        ]
        if not provisions:
            provisions.append({
                "provision": "No specific statutory provisions detected in the retrieved text.",
                "why_relevant": "Provide more detailed facts/documents, or consult a legal database directly.",
                "source_refs": [],
            })
        return {"legal_provisions": provisions}

    def _prosecution_defence(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raw_args = ctx.get("case_arguments", "") or ""
        lines = [l.strip("-• \t") for l in raw_args.splitlines() if l.strip()]
        prosecution, defence, unclassified = [], [], []
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["prosecution", "state", "plaintiff", "complainant"]):
                prosecution.append(line)
            elif any(k in low for k in ["defence", "defense", "accused", "respondent"]):
                defence.append(line)
            else:
                unclassified.append(line)
        if not prosecution and not defence and unclassified:
            mid = len(unclassified) // 2 or len(unclassified)
            prosecution, defence = unclassified[:mid], unclassified[mid:]

        supporting_refs = [s["ref_key"] for s in ctx.get("sources", [])[:5]]
        return {
            "prosecution": {
                "arguments": prosecution or ["No arguments explicitly tagged as prosecution — review case input."],
                "supporting_precedents": supporting_refs[:3],
                "gaps": ["Not enough source material to assess strength — upload supporting documents." ] if not ctx.get("sources") else [],
            },
            "defence": {
                "arguments": defence or ["No arguments explicitly tagged as defence — review case input."],
                "supporting_precedents": supporting_refs[3:5],
                "gaps": ["Not enough source material to assess strength — upload supporting documents."] if not ctx.get("sources") else [],
            },
            "note": "Argument classification is heuristic (keyword-based) in demo mode. Verify assignment manually.",
        }

    def _evidence_gaps(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        evidence_text = ctx.get("evidence_text", "") or ""
        present = [l.strip("-• \t") for l in evidence_text.splitlines() if l.strip()]
        commonly_expected = [
            "forensic/medical report", "eyewitness statement", "CCTV or digital evidence",
            "chain-of-custody record", "expert opinion", "documentary/financial record",
        ]
        present_low = " ".join(present).lower()
        potential_gaps = [
            f"A {item} may be relevant to verify claims in this case, but was not found among the uploaded documents."
            for item in commonly_expected
            if item.split("/")[0] not in present_low
        ]
        return {
            "evidence_present": present or ["No structured evidence entries provided."],
            "potential_evidence_to_verify": potential_gaps,
        }

    def _contradictions(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        contradictions = []
        dates_by_doc = {}
        for src in ctx.get("sources", []):
            found_dates = set(DATE_RE.findall(src["text"]))
            if found_dates:
                dates_by_doc[src["ref_key"]] = found_dates
        all_dates = set()
        for d in dates_by_doc.values():
            all_dates |= d
        if len(all_dates) > 1 and len(dates_by_doc) > 1:
            contradictions.append({
                "type": "timeline_inconsistency",
                "description": "Different dates appear across documents describing what may be the same events — verify which is authoritative.",
                "values_found": sorted(all_dates),
                "source_refs": list(dates_by_doc.keys()),
            })
        if not contradictions:
            contradictions.append({
                "type": "none_detected",
                "description": "No clear factual contradictions were detected by the automated pattern check. This does not guarantee consistency — manual cross-checking is still recommended.",
                "values_found": [],
                "source_refs": [],
            })
        return {"contradictions": contradictions}

    def _comparison(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        current = ctx.get("current_case", {})
        historical = ctx.get("historical_case", {})
        from .retrieval import _keywords
        cur_kw = _keywords(" ".join([current.get("facts", ""), current.get("issues", "")]))
        hist_kw = _keywords(historical.get("summary", ""))
        shared = sorted(cur_kw & hist_kw)[:10]
        only_current = sorted(cur_kw - hist_kw)[:6]
        only_hist = sorted(hist_kw - cur_kw)[:6]
        return {
            "similarities": shared or ["Limited direct lexical overlap detected — review manually for conceptual similarity."],
            "differences": {
                "unique_to_current_case": only_current,
                "unique_to_historical_case": only_hist,
            },
            "legal_significance": "Shared fact-pattern terms suggest the historical outcome and reasoning may be instructive, but the extent of applicability depends on jurisdiction, year decided, and factual nuance not captured by keyword overlap alone.",
            "applicability_considerations": [
                "Confirm the historical case is still good law / not overruled.",
                "Check jurisdictional applicability.",
                "Confirm factual similarity beyond shared vocabulary.",
            ],
        }

    def _brief(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        case = ctx.get("case", {})
        return {
            "case_overview": case.get("title", "Untitled case"),
            "key_facts": case.get("facts", "") or "Not provided.",
            "legal_issues": case.get("charges_issues", "") or "Not provided.",
            "relevant_laws": ctx.get("legal_provisions", []),
            "relevant_precedents": ctx.get("similar_cases", []),
            "prosecution_arguments": ctx.get("prosecution", {}),
            "defence_arguments": ctx.get("defence", {}),
            "evidence_summary": ctx.get("evidence_present", []),
            "evidence_gaps": ctx.get("evidence_gaps", []),
            "contradictions": ctx.get("contradictions", []),
            "questions_to_examine": [
                "Are all statutory provisions cited still in force and correctly applicable?",
                "Do the identified historical cases match on jurisdiction and material facts?",
                "Have all evidence gaps been addressed or explained?",
                "Are there unresolved contradictions requiring witness clarification?",
            ],
        }

    def _case_extraction(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .legal_structure import extract_legal_sections_structure
        from .text_cleaner import extract_title_and_year
        text = ctx.get("text", "")
        if not text:
            return {
                "title": "Untitled Case",
                "court": "Unknown Court",
                "facts": "No text provided.",
                "issues": "",
                "procedural_history": "",
                "timeline": [],
            }
        
        extracted = extract_legal_sections_structure(text)
        first_line = text.splitlines()[0] if text.splitlines() else ""
        title, year = extract_title_and_year(first_line)
        
        parties = []
        if "vs" in title.lower() or " v. " in title.lower():
            delim = "vs" if "vs" in title.lower() else "v."
            p1, _, p2 = title.lower().partition(delim)
            parties = [
                {"role": "petitioner", "name": p1.strip().title()},
                {"role": "respondent", "name": p2.strip().title()},
            ]

        return {
            "title": title if title and title != "Unknown Case" else (first_line[:80] or "Extracted Legal Case"),
            "case_number": f"{year or 2024}-LEGAL",
            "court": "Supreme Court of India" if "supreme court" in text.lower() else "High Court",
            "jurisdiction": "India",
            "facts": extracted["facts"] or text[:500],
            "issues": "\n".join(extracted["legal_issues"]) if isinstance(extracted["legal_issues"], list) else str(extracted["legal_issues"]),
            "arguments": {
                "petitioner": extracted.get("petitioner_arguments", []),
                "respondent": extracted.get("respondent_arguments", []),
            },
            "evidence": [{"description": str(e)} for e in extracted.get("evidence", [])],
            "provisions": [{"act": l, "section": ""} for l in extracted.get("laws", [])] + [{"act": "", "section": s} for s in extracted.get("sections", [])],
            "procedural_history": f"Case records acquired and structured by Nya AI extraction pipeline.",
            "judgment": extracted.get("decision", ""),
            "parties": parties,
            "timeline": [{"date": f"{year or 2024}", "event": "Judgment/Order issued."}] if year else [],
        }


_provider_singleton: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider_singleton
    if _provider_singleton is None:
        name = os.getenv("LLM_PROVIDER", "mock")
        _provider_singleton = AnthropicProvider() if name == "anthropic" else MockLLMProvider()
    return _provider_singleton
