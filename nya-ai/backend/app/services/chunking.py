"""Document chunking.

Splits on paragraph boundaries first, then packs paragraphs into
~CHUNK_SIZE-character windows with overlap, so retrieval units stay
semantically coherent instead of cutting mid-sentence. Tracks page number
when the source text has [[PAGE n]] markers inserted by text_extraction.
"""
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
    # Determine, for every character offset, which page we're on.
    page_marks = [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(full_text)]
    cleaned = PAGE_MARKER_RE.sub("", full_text)

    def page_for_offset(offset_in_cleaned: int) -> Optional[int]:
        if not page_marks:
            return None
        # crude but adequate mapping: walk page marks in original text order
        current = None
        consumed = 0
        for i, (pos, pnum) in enumerate(page_marks):
            next_pos = page_marks[i + 1][0] if i + 1 < len(page_marks) else len(full_text)
            span_len = max(next_pos - pos, 0)
            if consumed <= offset_in_cleaned < consumed + span_len:
                current = pnum
                break
            consumed += span_len
        return current if current is not None else page_marks[-1][1]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= CHUNK_SIZE:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
                overlap_tail = buf[-CHUNK_OVERLAP:]
                buf = f"{overlap_tail}\n\n{para}".strip()
            else:
                # single paragraph longer than CHUNK_SIZE: hard-split it
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunks.append(para[i:i + CHUNK_SIZE])
                buf = ""
    if buf:
        chunks.append(buf)

    if not chunks and cleaned.strip():
        chunks = [cleaned.strip()[:CHUNK_SIZE]]

    result = []
    offset = 0
    for i, c in enumerate(chunks):
        # We estimate the original character offset by finding the chunk text in the cleaned text
        start_idx = cleaned.find(c, offset)
        if start_idx == -1:
            start_idx = offset
            
        result.append(TextChunk(
            index=i, 
            text=c, 
            page_number=page_for_offset(start_idx),
            char_start=start_idx,
            char_end=start_idx + len(c)
        ))
        offset = start_idx + len(c)
    return result
