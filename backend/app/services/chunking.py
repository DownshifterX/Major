import re
from dataclasses import dataclass
from typing import List, Optional

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
PAGE_MARKER_RE = re.compile(r"\[\[PAGE (\d+)\]\]")

@dataclass
class TextChunk:
    index: int
    text: str
    page_number: Optional[int]
    section_type: str = "body"
    char_start: int = 0
    char_end: int = 0

def chunk_text(full_text: str) -> List[TextChunk]:
    page_marks = [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(full_text)]
    cleaned = PAGE_MARKER_RE.sub("", full_text)

    def page_for_offset(offset: int) -> Optional[int]:
        if not page_marks: return None
        cur, consumed = None, 0
        for i, (pos, pnum) in enumerate(page_marks):
            next_pos = page_marks[i + 1][0] if i + 1 < len(page_marks) else len(full_text)
            span_len = max(next_pos - pos, 0)
            if consumed <= offset < consumed + span_len:
                cur = pnum; break
            consumed += span_len
        return cur if cur is not None else page_marks[-1][1]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= CHUNK_SIZE:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
                buf = f"{buf[-CHUNK_OVERLAP:]}\n\n{para}".strip()
            else:
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunks.append(para[i:i + CHUNK_SIZE])
                buf = ""
    if buf: chunks.append(buf)
    if not chunks and cleaned.strip(): chunks = [cleaned.strip()[:CHUNK_SIZE]]

    result, offset = [], 0
    for i, c in enumerate(chunks):
        start = cleaned.find(c, offset)
        if start == -1: start = offset
        result.append(TextChunk(index=i, text=c, page_number=page_for_offset(start), char_start=start, char_end=start + len(c)))
        offset = start + len(c)
    return result
