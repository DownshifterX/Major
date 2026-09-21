"""Text cleaner and preprocessor specifically designed for Indian legal judgments.

Preserves legal structure, statutes, sections, citations, party names, and paragraphs
while stripping HTML tags, redundant navigation, Indian Kanoon headers/watermarks,
and normalizing whitespace and unicode.
"""
import re
from typing import Tuple

BOILERPLATE_PATTERNS = [
    r"Take your legal practice online with Kanoon\b.*",
    r"Try out our Premium Member services today\b.*",
    r"User Queries\b.*",
    r"Indian Kanoon - http://indiankanoon\.org/doc/\d+/?",
    r"http://indiankanoon\.org/doc/\d+/?",
    r"Get this document in PDF\b.*",
    r"Print this document\b.*",
    r"Powered by:?\s*National Informatics Centre\b.*",
    r"Disclaimer:?\s*Information and inputs.*",
]

COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


def clean_legal_text(raw_text: str) -> str:
    """Cleans legal judgment text while preserving substantive legal content."""
    if not raw_text:
        return ""

    text = raw_text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"<[^>]+>", " ", text)

    cleaned_lines = []
    for line in text.split("\n"):
        line_stripped = line.strip()
        is_boilerplate = any(bp.search(line_stripped) for bp in COMPILED_BOILERPLATE)
        if not is_boilerplate:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_title_and_year(title_str: str) -> Tuple[str, int | None]:
    """Extracts clean case name and judgment year from standard title string."""
    if not title_str:
        return ("Unknown Case", None)
    
    match = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,\s+(\d{4}))\b", title_str, re.IGNORECASE)
    year = None
    clean_title = title_str
    if match:
        clean_title = title_str[:match.start()].strip()
        try:
            year = int(match.group(2))
        except ValueError:
            pass
    else:
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", title_str)
        if year_match:
            year = int(year_match.group(1))

    clean_title = re.sub(r"\s+", " ", clean_title).strip()
    return (clean_title, year)
