"""Legal Structure and Entity Extraction Pipeline.

Extracts structured representation of legal cases:
- Facts
- Legal Issues
- Petitioner Arguments
- Respondent Arguments
- Evidence
- Laws & Statutory Sections (e.g. IPC, CrPC, CPC, Constitution)
- Precedents / Citations
- Reasoning
- Decision / Order / Relief

Also extracts named legal entities: judges, courts, sections, statutes, parties.
"""
import re
from typing import Dict, Any, List

# Indian statutory sections: Section 302 IPC, Sec. 482 of CrPC, Article 21, etc.
SECTION_PATTERN = re.compile(
    r"\b((?:Section|Sec\.?|Article|Art\.?|Rule|Order)\s+\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))?"
    r"(?:\s*(?:of|,)\s*(?:the\s*)?[A-Z][A-Za-z, ]{2,50}?(?:Act|Code|Constitution)(?:,?\s*\d{4})?)?)\b",
    re.IGNORECASE,
)

ACT_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z\s]{2,45}\s+(?:Act|Code|Rules|Regulations|Constitution)(?:,?\s*\d{4})?)\b"
)

CITATION_PATTERN = re.compile(
    r"\b(\d{4}\s+(?:SCC|AIR|SCR|SCALE|ILR|CrLJ)\s+(?:\([A-Za-z]+\)\s+)?\d+)\b"
)

BENCH_PATTERN = re.compile(
    r"\b(?:Hon['’]ble\s+)?(?:Mr\.|Mrs\.|Ms\.|Justice)\s+([A-Z][A-Za-z\.\s]{2,30}?)(?:,\s*J\.|\s+J\b|\s+Judge)",
    re.IGNORECASE,
)


def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    """Extracts named legal entities from text."""
    if not text:
        return {"sections": [], "laws": [], "citations": [], "judges": []}

    sections = []
    for m in SECTION_PATTERN.finditer(text):
        s = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(s) > 4 and s not in sections:
            sections.append(s)

    laws = []
    for m in ACT_PATTERN.finditer(text):
        l = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(l) > 6 and l not in laws and not l.startswith(("Section", "Article", "Order")):
            laws.append(l)

    citations = []
    for m in CITATION_PATTERN.finditer(text):
        c = m.group(1).strip()
        if c not in citations:
            citations.append(c)

    judges = []
    for m in BENCH_PATTERN.finditer(text):
        j = m.group(1).strip()
        if len(j) > 3 and j not in judges:
            judges.append(j)

    return {
        "sections": sections[:25],
        "laws": laws[:15],
        "citations": citations[:20],
        "judges": judges[:10],
    }


def extract_legal_sections_structure(text: str) -> Dict[str, Any]:
    """Segment-aware extraction of the canonical legal sections.
    
    Splits the judgment or case brief into:
    - facts
    - legal_issues
    - petitioner_arguments
    - respondent_arguments
    - evidence
    - reasoning
    - decision
    - relief
    """
    if not text:
        return {
            "facts": "",
            "legal_issues": [],
            "petitioner_arguments": [],
            "respondent_arguments": [],
            "evidence": [],
            "reasoning": [],
            "decision": "",
            "relief": "",
            "entities": {"sections": [], "laws": [], "citations": [], "judges": []},
        }

    entities = extract_legal_entities(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    facts_paras = []
    issues_paras = []
    pet_args = []
    resp_args = []
    evidence_items = []
    reasoning_paras = []
    decision_paras = []
    relief_paras = []

    current_section = "facts"

    for p in paragraphs:
        p_lower = p.lower()
        
        # Section boundary heuristic detectors
        if re.search(r"\b(issue|question(s)? for (our )?consideration|point(s)? for determination)\b", p_lower):
            current_section = "issues"
        elif re.search(r"\b(learned counsel for the (petitioner|appellant)|submitted (by|on behalf of) the (appellant|petitioner)|(petitioner|appellant)('s)? case|argument(s)? of the (appellant|petitioner)|contention of the (appellant|petitioner))\b", p_lower):
            current_section = "petitioner_arguments"
        elif re.search(r"\b(learned counsel for the (respondent|state)|counsel for the state|opposed by the (respondent|state)|on behalf of the (respondent|state)|state submitted)\b", p_lower):
            current_section = "respondent_arguments"
        elif re.search(r"\b(exhibit|pw-|dw-|witness|deposition|seizure memo|post-mortem report|forensic|cctv)\b", p_lower) and len(p) < 300:
            current_section = "evidence"
        elif re.search(r"\b(we have considered|in our opinion|in our considered view|it is clear that|we find that|discussion|analysis)\b", p_lower):
            current_section = "reasoning"
        elif re.search(r"\b(in the result|ordered accordingly|accordingly, the appeal|stands dismissed|stands allowed|conviction is upheld|we pass the following order)\b", p_lower):
            current_section = "decision"

        # Bucket paragraph
        if current_section == "facts":
            facts_paras.append(p)
        elif current_section == "issues":
            issues_paras.append(p)
        elif current_section == "petitioner_arguments":
            pet_args.append(p)
        elif current_section == "respondent_arguments":
            resp_args.append(p)
        elif current_section == "evidence":
            evidence_items.append(p)
        elif current_section == "reasoning":
            reasoning_paras.append(p)
        elif current_section == "decision":
            decision_paras.append(p)

    # Fallback consolidation if sections didn't trigger explicitly
    if not decision_paras and len(paragraphs) > 2:
        decision_paras = paragraphs[-2:]
    if not facts_paras and paragraphs:
        facts_paras = paragraphs[:3]

    return {
        "facts": "\n\n".join(facts_paras[:8]),
        "legal_issues": issues_paras[:5] if issues_paras else ["Determination of liability and compliance with statutory provisions."],
        "petitioner_arguments": pet_args[:6],
        "respondent_arguments": resp_args[:6],
        "evidence": evidence_items[:8],
        "reasoning": reasoning_paras[:10],
        "decision": "\n\n".join(decision_paras[:3]),
        "relief": decision_paras[-1] if decision_paras else "",
        "laws": entities["laws"],
        "sections": entities["sections"],
        "precedents": entities["citations"],
        "entities": entities,
    }
