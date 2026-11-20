import json, os, re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import httpx

STATUTE_RE = re.compile(
    r"\b((?:Section|Sec\.?|Article|Art\.?|Rule)\s+\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))?"
    r"(?:\s+(?:of|,)\s+the\s+[A-Z][A-Za-z, ]{2,60}?(?:Act|Code|Constitution)(?:,?\s*\d{4})?)?)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)

def _parse_strict_json(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return {"error": "Invalid JSON response", "raw_output": raw[:1000]}

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: pass
    @abstractmethod
    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]: pass

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set but LLM_PROVIDER=gemini")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def generate(self, prompt: str) -> str:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}}
        try:
            with httpx.Client(timeout=12.0) as client:
                resp = client.post(self.base_url, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if self.model != "gemini-3.5-flash-lite":
                try:
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={self.api_key}"
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.post(fallback_url, json=body)
                        resp.raise_for_status()
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception: pass
            raise RuntimeError(f"Gemini API generation error: {e}")

    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        full_prompt = f"{prompt}\n\nRespond with STRICT JSON only. No markdown fences, no preamble, no commentary."
        try:
            raw = self.generate(full_prompt)
            res = _parse_strict_json(raw)
            if not isinstance(res, dict) or "error" in res:
                return MockLLMProvider().generate_structured(prompt, context, analysis_type)
            return res
        except Exception:
            return MockLLMProvider().generate_structured(prompt, context, analysis_type)

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set but LLM_PROVIDER=anthropic")

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": self.model, "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            return "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")

    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        return _parse_strict_json(self.generate(f"{prompt}\n\nRespond with STRICT JSON only."))

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "[mock provider] Offline deterministic rule-based output."

    def generate_structured(self, prompt: str, context: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        handlers = {
            "similar_cases": self._similar_cases, "legal_provisions": self._legal_provisions,
            "prosecution_defence": self._prosecution_defence, "evidence_gaps": self._evidence_gaps,
            "contradictions": self._contradictions, "comparison": self._comparison,
            "brief": self._brief, "case_extraction": self._case_extraction,
        }
        h = handlers.get(analysis_type)
        return h(context) if h else {"error": f"No mock handler for '{analysis_type}'"}

    def _similar_cases(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        res = []
        if ctx.get("retrieved_groups"):
            for g in ctx["retrieved_groups"]:
                hc, best = g["historical_case"], g["chunks"][0]
                res.append({
                    "case_name": hc["case_name"], "court": hc.get("court"), "year": hc.get("year"),
                    "relevance_score": round(best.get("hybrid_score", 0.5), 3),
                    "why_relevant": self._why_relevant(ctx.get("query_text", ""), best["text"]),
                    "matching_excerpt": best["text"][:400], "ref_key": best.get("ref_key", "S1"),
                })
        elif ctx.get("sources"):
            seen = set()
            for s in ctx["sources"]:
                cname = s.get("case_name") or "Precedent Case"
                if cname in seen or cname == "current case document": continue
                seen.add(cname)
                res.append({
                    "case_name": cname, "court": s.get("court") or "High Court", "year": s.get("year"),
                    "relevance_score": round(s.get("hybrid_score") or 0.65, 3),
                    "why_relevant": self._why_relevant(ctx.get("query_text", ""), s.get("text", "")),
                    "matching_excerpt": (s.get("text") or "")[:400], "ref_key": s.get("ref_key", "S1"),
                })
        return {"similar_cases": res}

    def _why_relevant(self, query: str, text: str) -> str:
        from .retrieval import _keywords
        shared = sorted(_keywords(query) & _keywords(text))[:6]
        return f"Shares key terms/fact-patterns: {', '.join(shared)}." if shared else "Retrieved as closest available match in corpus."

    def _legal_provisions(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        found = {}
        for src in ctx.get("sources", []):
            for m in STATUTE_RE.finditer(src.get("text", "")):
                found.setdefault(re.sub(r"\s+", " ", m.group(1)).strip(), []).append(src["ref_key"])
        return {"legal_provisions": [{"provision": k, "why_relevant": "Referenced in retrieved sources.", "source_refs": sorted(set(v))} for k, v in found.items()]}

    def _clean_arg(self, t: str) -> str:
        t = re.sub(r":::\s*Downloaded on\b.*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*\d+\s+of\s+\d+\s*$", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\[?\d+(?:\.\d+)?\]?\s*", "", t.strip())
        return " ".join(t.split())

    def _prosecution_defence(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        args = {"petitioner": [], "respondent": []}
        for s in ctx.get("sources", []):
            txt = s.get("text", "")
            for line in txt.split("\n"):
                cl = self._clean_arg(line)
                if len(cl) > 30:
                    l_low = line.lower()
                    if any(k in l_low for k in ["petitioner", "appellant", "plaintiff", "claimant"]):
                        args["petitioner"].append(f"{cl} [{s['ref_key']}]")
                    elif any(k in l_low for k in ["respondent", "defendant", "state"]):
                        args["respondent"].append(f"{cl} [{s['ref_key']}]")

        c_args = ctx.get("case_arguments", {})
        if isinstance(c_args, dict):
            for p in c_args.get("petitioner", [])[:4]:
                cp = self._clean_arg(p)
                if len(cp) > 20: args["petitioner"].append(cp)
            for r in c_args.get("respondent", [])[:4]:
                cr = self._clean_arg(r)
                if len(cr) > 20: args["respondent"].append(cr)

        refs = [s["ref_key"] for s in ctx.get("sources", [])]
        p_args = list(dict.fromkeys(args["petitioner"]))[:6] or ["Petitioner claims statutory relief under applicable land acts."]
        r_args = list(dict.fromkeys(args["respondent"]))[:6] or ["Respondent contends proper administrative procedure was followed."]
        return {
            "prosecution": {"arguments": p_args, "supporting_precedents": refs[:3], "gaps": ["Independent valuation report not attached."]},
            "defence": {"arguments": r_args, "supporting_precedents": refs[3:6], "gaps": ["Notice service acknowledgment missing."]}
        }

    def _evidence_gaps(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        present = []
        for s in ctx.get("sources", []):
            for d in DATE_RE.findall(s.get("text", "")):
                present.append(f"Record with date {d} [{s['ref_key']}]")
        return {
            "evidence_present": list(dict.fromkeys(present))[:8] or ["Documentary pleadings on record."],
            "potential_evidence_to_verify": ["Certified mutation records", "Survey measurement plan", "Collector compensation award"]
        }

    def _contradictions(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        dates = {}
        for s in ctx.get("sources", []):
            for d in DATE_RE.findall(s.get("text", "")):
                dates.setdefault(d, []).append(s["ref_key"])
        contras = []
        if len(dates) >= 2:
            items = list(dates.items())
            contras.append({
                "type": "timeline_conflict",
                "description": f"Multiple critical dates recorded across pleadings: '{items[0][0]}' vs '{items[1][0]}'.",
                "source_refs": sorted(set(items[0][1] + items[1][1]))
            })
        return {"contradictions": contras}

    def _comparison(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        srcs = ctx.get("sources", [])
        refs = [s["ref_key"] for s in srcs]
        return {
            "similarities": [f"Both cases involve determination of land rights and compensation [{','.join(refs[:2])}]."] if refs else ["Similar factual domain."],
            "differences": {"unique_to_current_case": ["Specific survey number and collector award amount."], "unique_to_historical_case": ["Prior appellate bench ruling."]},
            "legal_significance": "Illustrates statutory compensation principles under Land Acquisition Act.",
            "applicability_considerations": ["Verify whether later legislative amendments apply to the current notification date."]
        }

    def _brief(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        case = ctx.get("case", {})
        prior = ctx.get("prior_analyses", {})
        pd = prior.get("prosecution_defence", {})
        eg = prior.get("evidence_gaps", {})
        return {
            "case_overview": f"Summary brief for {case.get('title', 'Case')}. Court: {case.get('court', 'High Court')}.",
            "key_facts": case.get("facts", "")[:500] or "Facts pending review.",
            "legal_issues": case.get("issues", "")[:300] or "Issues under adjudication.",
            "relevant_laws": [p.get("provision") for p in prior.get("legal_provisions", {}).get("legal_provisions", [])[:5]],
            "relevant_precedents": [sc.get("case_name") for sc in prior.get("similar_cases", {}).get("similar_cases", [])[:5]],
            "prosecution_arguments": pd.get("prosecution", {}),
            "defence_arguments": pd.get("defence", {}),
            "evidence_summary": eg.get("evidence_present", []),
            "evidence_gaps": eg.get("potential_evidence_to_verify", []),
            "contradictions": [c.get("description") for c in prior.get("contradictions", {}).get("contradictions", [])],
            "questions_to_examine": ["Whether reference court applied correct deduction on exemplar sale deeds?", "Whether statutory notice was properly served?"]
        }

    def _case_extraction(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        from .legal_structure import extract_legal_sections_structure
        from .text_cleaner import extract_title_and_year
        txt = ctx.get("text", "")
        extracted = extract_legal_sections_structure(txt)
        title, year = extract_title_and_year(txt[:500])
        return {
            "title": title or "Acquired Legal Case", "case_number": "", "court": "High Court",
            "jurisdiction": "Appellate Civil", "judge": "", "date_of_judgment": f"{year or 2024}",
            "date_of_filing": "", "case_type": "Land Acquisition", "legal_domain": "Land & Property Law",
            "facts": extracted.get("facts", ""), "issues": "\n".join(extracted.get("legal_issues", [])),
            "arguments": {"petitioner": extracted.get("petitioner_arguments", []), "respondent": extracted.get("respondent_arguments", [])},
            "evidence": [{"description": str(e)} for e in extracted.get("evidence", [])],
            "provisions": [{"act": l, "section": ""} for l in extracted.get("laws", [])] + [{"act": "", "section": s} for s in extracted.get("sections", [])],
            "procedural_history": "Structured by Nya AI extraction pipeline.", "judgment": extracted.get("decision", ""),
            "parties": [], "timeline": [{"date": f"{year or 2024}", "event": "Judgment/Order issued."}] if year else []
        }

_provider_instance: Optional[LLMProvider] = None

def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance
    kind = os.getenv("LLM_PROVIDER", "mock").lower()
    if kind == "gemini":
        try: _provider_instance = GeminiProvider()
        except Exception: _provider_instance = MockLLMProvider()
    elif kind == "anthropic":
        _provider_instance = AnthropicProvider()
    else:
        _provider_instance = MockLLMProvider()
    return _provider_instance
