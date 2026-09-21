from sqlalchemy.orm import Session
from .. import models
from .chunking import chunk_text
from .embeddings import get_embedding_provider


def index_document(db: Session, document: models.Document) -> int:
    """Chunk a document's full_text, embed the chunks, and persist them.

    Returns the number of chunks created. Idempotent: re-running clears any
    previously stored chunks for this document first.
    """
    db.query(models.Chunk).filter(models.Chunk.document_id == document.id).delete()

    pieces = chunk_text(document.full_text or "")
    if not pieces:
        document.num_chunks = 0
        db.commit()
        return 0

    provider = get_embedding_provider()
    vectors = provider.embed([p.text for p in pieces])

    for piece, vector in zip(pieces, vectors):
        chunk = models.Chunk(
            document_id=document.id,
            case_id=document.case_id,
            historical_case_id=document.historical_case_id,
            chunk_index=piece.index,
            page_number=piece.page_number,
            char_start=piece.char_start,
            char_end=piece.char_end,
            section_type=getattr(piece, "section_type", "body"),
            text=piece.text,
            embedding=vector,
        )
        db.add(chunk)

    document.num_chunks = len(pieces)
    db.commit()
    return len(pieces)
