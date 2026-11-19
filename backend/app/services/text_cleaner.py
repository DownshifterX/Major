import re
from typing import Tuple

BOILERPLATE_PATTERNS = [
    r"Take your legal practice online with Kanoon\b.*", r"Try out our Premium Member services today\b.*",
    r"User Queries\b.*", r"https?://indiankanoon\.org/doc/\d+/?", r"Get this document in PDF\b.*",
    r"Print this document\b.*", r"Powered by:?\s*National Informatics Centre\b.*", r"Disclaimer:?\s*Information and inputs.*",
    r"^\s*\[\s*Cites\s*\d*\s*,\s*Cited by\s*\d*\s*\]\s*$", r"^\s*\[\s*Cites\s*$", r"^\s*,\s*Cited by\s*$",
    r"^\s*Signature Not Verified\s*$", r"^\s*Digitally signed by\b.*", r"^\s*Date:\s*\d{4}\.\d{2}\.\d{2}.*",
    r"^\s*Reason:\s*$", r".*Page\s+\d+\s+of\s+\d+\s*$", r"^\s*\d+\s+of\s+\d+\s*$", r"^\s*\d+\s*/\s*\d+\s*$",
    r":::\s*Downloaded on\b.*", r".*RFA No\.\s*\d+.*connected cases.*",
]
COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

def clean_legal_text(raw_text: str) -> str:
    if not raw_text: return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*\[\s*Cites[\s\S]*?\]\s*", "", text, flags=re.IGNORECASE)

    cleaned_lines = [line for line in text.split("\n") if not any(bp.search(line.strip()) for bp in COMPILED_BOILERPLATE)]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    return re.sub(r"[ \t]{2,}", " ", text).strip()

def extract_title_and_year(title_str: str) -> Tuple[str, int | None]:
    if not title_str: return "", None
    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", title_str)
    year = int(m_year.group(1)) if m_year else None
    cleaned_title = re.sub(r"\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}", "", title_str, flags=re.IGNORECASE)
    cleaned_title = re.sub(r"\s+vs\.?\s+", " vs ", cleaned_title, flags=re.IGNORECASE)
    return cleaned_title.strip(), year
