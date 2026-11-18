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
    """Checks whether the given case text or type relates to land and property law."""
    combined = f"{case_type} {text}".strip()
    if not combined or len(combined) < 25:
        return True
    if NON_LAND_KEYWORDS.search(combined) and not LAND_KEYWORDS.search(combined):
        return False
    return True


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


# Specific evidence pattern for documentary, witness, and physical exhibits in Indian judgments
EVIDENCE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:Exs?\b\.?|Exhibit\b)\s*[-–—]?\s*[A-Z0-9\-\/]+"
    r"|(?:PW|DW|CW)\s*[-–—]?\s*\d+\b(?:\s*\([^\)\n]{2,35}\))?"
    r"|(?:registered\s+)?(?:sale\s+deed|relinquishment\s+deed|gift\s+deed|mortgage\s+deed|partition\s+deed|title\s+deed|will|power\s+of\s+attorney)\b"
    r"|(?:khatian|patta|revenue\s+records?|mutation\s+entry|jamabandi|khasra(?:\s+girdawari)?|khatauni)\b"
    r"|(?:certified\s+copy|certified\s+copies)\s+of\s+[^,\.\n]{3,40}"
    r"|(?:seizure\s+memo|post-mortem\s+report|forensic\s+report|fsl\s+report)\b"
    r")",
    re.IGNORECASE,
)


def extract_evidence_items(text: str) -> List[str]:
    """Extracts cleanly identified documentary, witness, and physical evidence items."""
    if not text:
        return []
    items = []
    seen = set()
    for m in EVIDENCE_PATTERN.finditer(text):
        val = " ".join(m.group(0).split()).strip()
        # Filter trivial false positives
        if len(val) > 2 and val.lower() not in seen and not val.lower().startswith(("exercise", "exist", "express")):
            seen.add(val.lower())
            items.append(val)
    return items[:15]


def extract_legal_sections_structure(text: str, kanoon_structure: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """Segment-aware extraction of canonical legal sections.
    
    Extracts:
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
            "laws": [],
            "sections": [],
            "precedents": [],
            "entities": {"sections": [], "laws": [], "citations": [], "judges": []},
        }

    entities = extract_legal_entities(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # If Indian Kanoon data-structure tags are available, leverage them directly
    kanoon = kanoon_structure or {}
    k_facts = kanoon.get("Facts", [])
    k_issues = kanoon.get("Issue", [])
    k_pet = kanoon.get("PetArg", [])
    k_resp = kanoon.get("RespArg", [])
    k_decision = kanoon.get("Conclusion", [])

    facts_paras = []
    issues_paras = []
    pet_args = []
    resp_args = []
    reasoning_paras = []
    decision_paras = []

    # Filter out initial header/metadata lines and page numbers from substantive paragraphs
    substantive_paras = []
    for p in paragraphs:
        p_clean = p.strip()
        # Skip pure court headers, page footers, or short date/signer blocks
        if re.search(r"^(?:\[\s*Cites|Patna High Court|Supreme Court of India|IN THE HIGH COURT|Author:|Bench:|\d{4}\s+INSC|\s*C\.A\.\s*@|Signature Not Verified|Digitally signed|Date:\s*\d|Reason:|CIVIL APPEAL|JUDGMENT|NON-REPORTABLE|CORAM:|VERSUS)", p_clean, re.I):
            continue
        if re.search(r"^(?:[A-Z\s]{3,30},\s*\d{4}\.?|\s*Page\s+\d+\s+of\s+\d+|\s*\d+\s*/\s*\d+\s*)$", p_clean, re.I):
            continue
        if re.search(r"\b(?:APPELLANT|RESPONDENT|PETITIONER)\(?S?\)?\s*$", p_clean, re.I):
            continue
        if re.search(r"^[A-Za-z\s\.,&]+ vs?\.? [A-Za-z\s\.,&]+ on \d{1,2} [A-Za-z]+, \d{4}$", p_clean, re.I):
            continue
        if re.search(r"^(?:IN THE SUPREME COURT|CIVIL APPELLATE)", p_clean, re.I):
            continue
        if len(p_clean) < 18 or p_clean.startswith(('1. Leave granted.', 'Leave granted.')):
            continue
        substantive_paras.append(p_clean)

    if not substantive_paras:
        substantive_paras = paragraphs

    current_section = "facts"

    for p in substantive_paras:
        p_lower = p.lower()
        
        # Section boundary heuristic detectors with precise word boundaries
        if re.search(r"\b(issues?\s+for\s+(?:our\s+)?consideration|points?\s+for\s+determination|questions?\s+(?:arising|for\s+consideration)|following\s+issues?\s+arise|substantial\s+questions?\s+of\s+law)\b", p_lower):
            current_section = "issues"
            issues_paras.append(p)
            continue
        elif re.search(r"\b(?:learned\s+(?:counsel|advocate|senior\s+counsel)\s+(?:for|appearing\s+for|on\s+behalf\s+of)\s+the\s+(?:petitioner|petitioners|appellant|appellants|plaintiff|plaintiffs|claimant|claimants)|submitted\s+(?:by|on\s+behalf\s+of)\s+the\s+(?:appellant|appellants|petitioner|petitioners|plaintiff|plaintiffs)|(?:petitioner|petitioners|appellant|appellants|plaintiff|plaintiffs)(?:['’]s)?\s+(?:case|argument|arguments|contention|contentions|submission|submissions))\b", p_lower):
            current_section = "petitioner_arguments"
            pet_args.append(p)
            continue
        elif re.search(r"\b(?:learned\s+(?:counsel|advocate|senior\s+counsel)\s+(?:for|appearing\s+for|on\s+behalf\s+of)\s+the\s+(?:respondent|respondents|state|defendant|defendants|opposite\s+party)|counsel\s+for\s+the\s+state|opposed\s+by\s+the\s+(?:respondent|respondents|state|defendant|defendants)|(?:respondent|respondents|defendant|defendants|state)(?:['’]s)?\s+(?:counsel|argument|arguments|contention|contentions|submission|submissions)|state\s+submitted)\b", p_lower):
            current_section = "respondent_arguments"
            resp_args.append(p)
            continue
        elif re.search(r"\b(we\s+have\s+considered|in\s+our\s+opinion|in\s+our\s+considered\s+view|it\s+is\s+clear\s+that|we\s+find\s+that|discussion|having\s+heard\s+learned\s+counsel)\b", p_lower):
            current_section = "reasoning"
            reasoning_paras.append(p)
            continue
        elif re.search(r"\b(in\s+the\s+result|ordered\s+accordingly|accordingly,\s+the\s+appeal|stands\s+dismissed|stands\s+allowed|appeal\s+is\s+allowed|appeal\s+is\s+dismissed|conviction\s+is\s+upheld|set\s+aside,\s+and\s+the\s+judgment|we\s+pass\s+the\s+following\s+order|the\s+writ\s+petition\s+is\s+(?:allowed|dismissed))\b", p_lower):
            current_section = "decision"
            decision_paras.append(p)
            continue

        # Bucket paragraph by current state
        if current_section == "facts":
            facts_paras.append(p)
        elif current_section == "issues":
            issues_paras.append(p)
        elif current_section == "petitioner_arguments":
            pet_args.append(p)
        elif current_section == "respondent_arguments":
            resp_args.append(p)
        elif current_section == "reasoning":
            reasoning_paras.append(p)
        elif current_section == "decision":
            decision_paras.append(p)

    # Use Kanoon tags when heuristic missed or needs enhancement
    final_facts = k_facts if k_facts else facts_paras[:8]
    final_issues = k_issues if k_issues else (issues_paras[:5] if issues_paras else ["Determination of title, possession rights, and statutory compliance under land laws."])
    final_pet = k_pet if k_pet else pet_args[:6]
    final_resp = k_resp if k_resp else resp_args[:6]
    
    # Filter decision paragraphs to only include substantive rulings (not signer blocks)
    substantive_decisions = [
        d for d in (decision_paras or substantive_paras[-3:])
        if not re.search(r"^(?:[A-Z\s\.]+\s*,\s*J\.?|\s*\[[A-Z\s]+\]|\s*[A-Z\s]{4,15}\s+\d{1,2},\s*\d{4}\.?)\s*$", d.strip())
        and len(d.strip()) > 25
    ]
    final_decision = k_decision if k_decision else (substantive_decisions[:3] if substantive_decisions else substantive_paras[-2:])

    # Extract distinct, structured evidence items across the entire substantive text
    evidence_extracted = extract_evidence_items(text)

    # Format facts cleanly
    facts_text = "\n\n".join(final_facts[:6]) if isinstance(final_facts, list) else str(final_facts)
    # Remove any lingering initial court headers or judge author signatures from facts
    facts_text = re.sub(r"^(?:Patna High Court[^\n]*\n+|Supreme Court of India[^\n]*\n+|Author:[^\n]*\n+|Bench:[^\n]*\n+|[A-Z\s\.]+\s*,\s*J\.?\s*\n+)+", "", facts_text).strip()

    return {
        "facts": facts_text,
        "legal_issues": final_issues[:5],
        "petitioner_arguments": [p.strip() for p in final_pet[:6] if p.strip()],
        "respondent_arguments": [p.strip() for p in final_resp[:6] if p.strip()],
        "evidence": evidence_extracted,
        "reasoning": reasoning_paras[:10],
        "decision": "\n\n".join(final_decision[:3]) if isinstance(final_decision, list) else str(final_decision),
        "relief": final_decision[-1] if (isinstance(final_decision, list) and final_decision) else "",
        "laws": entities["laws"],
        "sections": entities["sections"],
        "precedents": entities["citations"],
        "entities": entities,
    }
