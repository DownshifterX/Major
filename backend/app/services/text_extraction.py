import io
from pypdf import PdfReader
import docx

class UnsupportedFileType(Exception): pass

def extract_text(filename: str, raw_bytes: bytes) -> str:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [f"\n\n[[PAGE {i + 1}]]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
        full = "".join(pages)
        if len("\n".join([line for line in full.splitlines() if not line.strip().startswith("[[PAGE")]).strip()) < 50:
            raise ValueError("Unable to extract usable text. File may be image-only scan.")
        return full
    if fn.endswith((".txt", ".md")):
        return raw_bytes.decode("utf-8", errors="replace")
    if fn.endswith(".docx"):
        doc = docx.Document(io.BytesIO(raw_bytes))
        return "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    raise UnsupportedFileType(f"Unsupported file type for '{filename}'. Supported: .pdf, .txt, .md, .docx")
