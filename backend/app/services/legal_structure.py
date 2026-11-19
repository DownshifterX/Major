import re
from typing import Dict, Any, List

SECTION_PATTERN = re.compile(
    r"\b((?:Section|Sec\.?|Article|Art\.?|Rule|Order)\s+\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))?"
    r"(?:\s*(?:of|,)\s*(?:the\s*)?[A-Z][A-Za-z, ]{2,50}?(?:Act|Code|Constitution)(?:,?\s*\d{4})?)?)\b",
    re.IGNORECASE,
)
ACT_PATTERN = re.compile(r"\b([A-Z][A-Za-z\s]{2,45}\s+(?:Act|Code|Rules|Regulations|Constitution)(?:,?\s*\d{4})?)\b")
CITATION_PATTERN = re.compile(r"\b(\d{4}\s+(?:SCC|AIR|SCR|SCALE|ILR|CrLJ)\s+(?:\([A-Za-z]+\)\s+)?\d+)\b")
BENCH_PATTERN = re.compile(r"\b(?:Hon['’]ble\s+)?(?:Mr\.|Mrs\.|Ms\.|Justice)\s+([A-Z][A-Za-z\.\s]{2,30}?)(?:,\s*J\.|\s+J\b|\s+Judge)", re.IGNORECASE)

NON_LAND_KEYWORDS = re.compile(
    r"\b(murder|ipc|crpc|theft|robbery|cheating|divorce|matrimonial|custody|bail|dacoity|kidnapping|ransom|rape|assault)\b",
    re.IGNORECASE,
)
LAND_KEYWORDS = re.compile(
    r"\b(land|property|real estate|tenancy|zamindari|ceiling|acquisition|patta|khatian|boundary|"
    r"possession|ejectment|rent|lease|mortgage|partition|revenue|mutation|collector|tehsildar|"
    r"eviction|easement|encroachment|plot|survey|title dispute)\b",
    re.IGNORECASE,
)

def is_land_case(text: str = "", case_type: str = "") -> bool:
    comb = f"{case_type} {text}".strip()
    return not (len(comb) >= 25 and NON_LAND_KEYWORDS.search(comb) and not LAND_KEYWORDS.search(comb))

def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"sections": [], "laws": [], "citations": [], "judges": []}
    def _find_dedup(pat, min_len=1, filter_fn=lambda x: True, limit=20):
        res, seen = [], set()
        for m in pat.finditer(text):
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(val) >= min_len and val not in seen and filter_fn(val):
                seen.add(val); res.append(val)
                if len(res) >= limit: break
        return res

    return {
        "sections": _find_dedup(SECTION_PATTERN, min_len=4, limit=25),
        "laws": _find_dedup(ACT_PATTERN, min_len=6, filter_fn=lambda x: not x.startswith(("Section", "Article", "Order")), limit=15),
        "citations": _find_dedup(CITATION_PATTERN, limit=20),
        "judges": _find_dedup(BENCH_PATTERN, min_len=3, limit=10),
    }

EVIDENCE_PATTERN = re.compile(
    r"\b((?:Exs?\b\.?|Exhibit\b)\s*[-–—]?\s*[A-Z0-9\-\/]+"
    r"|(?:PW|DW|CW)\s*[-–—]?\s*\d+\b(?:\s*\([^\)\n]{2,35}\))?"
    r"|(?:registered\s+)?(?:sale|relinquishment|gift|mortgage|partition|title)\s+deed|will|power\s+of\s+attorney"
    r"|khatian|patta|revenue\s+records?|mutation\s+entry|jamabandi|khasra(?:\s+girdawari)?|khatauni"
    r"|(?:certified\s+copy|certified\s+copies)\s+of\s+[^,\.\n]{3,40}"
    r"|seizure\s+memo|post-mortem\s+report|forensic\s+report|fsl\s+report)\b",
    re.IGNORECASE,
)

def extract_evidence_items(text: str) -> List[str]:
    if not text: return []
    items, seen = [], set()
    for m in EVIDENCE_PATTERN.finditer(text):
        val = " ".join(m.group(0).split()).strip()
        v_low = val.lower()
        if len(val) > 2 and v_low not in seen and not v_low.startswith(("exercise", "exist", "express")):
            seen.add(v_low); items.append(val)
            if len(items) >= 15: break
    return items

def extract_legal_sections_structure(text: str, kanoon_structure: Dict[str, List[str]] = None) -> Dict[str, Any]:
    empty_resp = {"facts": "", "legal_issues": [], "petitioner_arguments": [], "respondent_arguments": [],
                  "evidence": [], "reasoning": [], "decision": "", "relief": "", "laws": [], "sections": [],
                  "precedents": [], "entities": {}}
    if not text: return empty_resp

    entities = extract_legal_entities(text)
    ks = kanoon_structure or {}
    k_facts, k_issues = ks.get("facts", []), ks.get("issues", [])
    k_pet, k_resp = ks.get("petitioner_arguments", []), ks.get("respondent_arguments", [])
    k_decision = ks.get("decision", [])

    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    buckets: Dict[str, List[str]] = {
        "facts": [], "issues": [], "petitioner_arguments": [],
        "respondent_arguments": [], "reasoning": [], "decision": []
    }
    cur_sec = "facts"

    PATTERNS = [
        ("decision", r"\b(in\s+the\s+result|ordered\s+accordingly|accordingly,\s+the\s+appeal|stands\s+dismissed|stands\s+allowed|appeal\s+is\s+allowed|appeal\s+is\s+dismissed|conviction\s+is\s+upheld|set\s+aside,\s+and\s+the\s+judgment|we\s+pass\s+the\s+following\s+order|the\s+writ\s+petition\s+is\s+(?:allowed|dismissed))\b"),
        ("issues", r"\b(issues?\s+for\s+(?:our\s+)?consideration|points?\s+for\s+determination|questions?\s+(?:arising|for\s+consideration)|following\s+issues?\s+arise|substantial\s+questions?\s+of\s+law)\b"),
        ("petitioner_arguments", r"\b(learned\s+(?:counsel|advocate|senior\s+counsel)\s+(?:for|appearing\s+for|on\s+behalf\s+of)\s+the\s+(?:petitioner|petitioners|appellant|appellants|plaintiff|plaintiffs|claimant|claimants)|submitted\s+(?:by|on\s+behalf\s+of)\s+the\s+(?:appellant|appellants|petitioner|petitioners|plaintiff|plaintiffs)|(?:petitioner|petitioners|appellant|appellants|plaintiff|plaintiffs)(?:['’]s)?\s+(?:case|argument|arguments|contention|contentions|submission|submissions))\b"),
        ("respondent_arguments", r"\b(learned\s+(?:counsel|advocate|senior\s+counsel)\s+(?:for|appearing\s+for|on\s+behalf\s+of)\s+the\s+(?:respondent|respondents|state|defendant|defendants|opposite\s+party)|counsel\s+for\s+the\s+state|opposed\s+by\s+the\s+(?:respondent|respondents|state|defendant|defendants)|(?:respondent|respondents|defendant|defendants|state)(?:['’]s)?\s+(?:counsel|argument|arguments|contention|contentions|submission|submissions)|state\s+submitted)\b"),
        ("reasoning", r"\b(we\s+have\s+considered|in\s+our\s+opinion|in\s+our\s+considered\s+view|it\s+is\s+clear\s+that|we\s+find\s+that|discussion|having\s+heard\s+learned\s+counsel)\b"),
        ("facts", r"\b(brief\s+facts|factual\s+matrix|prosecution\s+case|facts\s+of\s+the\s+case|case\s+of\s+the\s+petitioner|case\s+of\s+the\s+appellant|briefly\s+stated|case\s+arises\s+out\s+of)\b"),
    ]

    for p in paras:
        p_low = p.lower()
        matched = False
        for sec, pat in PATTERNS:
            if re.search(pat, p_low):
                cur_sec = sec
                buckets[sec].append(p)
                matched = True
                break
        if not matched and cur_sec:
            buckets[cur_sec].append(p)

    final_facts = k_facts if k_facts else buckets["facts"][:8]
    final_issues = k_issues if k_issues else (buckets["issues"][:5] or ["Determination of title, possession rights, and statutory compliance under land laws."])
    final_pet = k_pet if k_pet else buckets["petitioner_arguments"][:6]
    final_resp = k_resp if k_resp else buckets["respondent_arguments"][:6]

    subs_dec = [d for d in (buckets["decision"] or paras[-3:]) if len(d.strip()) > 25 and not re.search(r"^(?:[A-Z\s\.]+\s*,\s*J\.?|\s*\[[A-Z\s]+\]|\s*[A-Z\s]{4,15}\s+\d{1,2},\s*\d{4}\.?)\s*$", d.strip())]
    final_dec = k_decision if k_decision else (subs_dec[:3] or paras[-2:])

    facts_text = "\n\n".join(final_facts[:6]) if isinstance(final_facts, list) else str(final_facts)
    facts_text = re.sub(r"^(?:Patna High Court[^\n]*\n+|Supreme Court of India[^\n]*\n+|Author:[^\n]*\n+|Bench:[^\n]*\n+|[A-Z\s\.]+\s*,\s*J\.?\s*\n+)+", "", facts_text).strip()

    return {
        "facts": facts_text,
        "legal_issues": final_issues[:5],
        "petitioner_arguments": [p.strip() for p in final_pet[:6] if p.strip()],
        "respondent_arguments": [p.strip() for p in final_resp[:6] if p.strip()],
        "evidence": extract_evidence_items(text),
        "reasoning": buckets["reasoning"][:10],
        "decision": "\n\n".join(final_dec[:3]) if isinstance(final_dec, list) else str(final_dec),
        "relief": final_dec[-1] if (isinstance(final_dec, list) and final_dec) else "",
        "laws": entities["laws"], "sections": entities["sections"],
        "precedents": entities["citations"], "entities": entities,
    }
