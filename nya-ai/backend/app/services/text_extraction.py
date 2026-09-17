"""Extracts plain text from uploaded documents.

Kept intentionally simple for the PoC: .txt/.md are read directly, .pdf is
parsed with pypdf, .docx with python-docx. Anything else is rejected with a clear error rather than
silently mishandled.
"""
import io
from pypdf import PdfReader
import docx


class UnsupportedFileType(Exception):
    pass


def extract_text(filename: str, raw_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(raw_bytes)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return raw_bytes.decode("utf-8", errors="replace")
    if lower.endswith(".docx"):
        return _extract_docx(raw_bytes)
    raise UnsupportedFileType(
        f"Unsupported file type for '{filename}'. Supported: .pdf, .txt, .md, .docx"
    )


def _extract_pdf(raw_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n\n[[PAGE {i + 1}]]\n{text}")
        
    full_text = "".join(pages)
    
    # Check if text extraction yielded anything meaningful (excluding our page markers)
    cleaned_text = "\n".join([line for line in full_text.splitlines() if not line.strip().startswith("[[PAGE")])
    if len(cleaned_text.strip()) < 50:
        raise ValueError("Unable to extract usable text from this document. It may be a scanned PDF or image-based file requiring OCR, which is not supported in this environment.")
        
    return full_text

def _extract_docx(raw_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(raw_bytes))
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n\n".join(text)
