"""Retrieval layer.

Combines:
  - semantic similarity (cosine distance between hashing-embeddings)
  - lexical overlap (shared distinctive keyword ratio)
into a single hybrid score, per RAG-quality requirements: retrieval should
find *legally relevant* material, not just textually similar words, and
provenance (document, case, page, chunk) must never be lost.

Deduplicates near-identical chunks and diversifies results across distinct
historical cases before returning the ranked list.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from .. import models
from .embeddings import get_embedding_provider, cosine_similarity

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were",
    "for", "by", "with", "that", "this", "it", "as", "at", "be", "are", "from",
    "has", "have", "had", "which", "who", "whom", "not", "but", "his", "her",
    "their", "its", "he", "she", "they", "will", "shall", "may", "such",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _keyword_overlap(query_kw: set, chunk_kw: set) -> float:
    if not query_kw or not chunk_kw:
        return 0.0
    inter = query_kw & chunk_kw
    return len(inter) / len(query_kw)


@dataclass
class RetrievedChunk:
    chunk: models.Chunk
    semantic_score: float
    keyword_score: float
    hybrid_score: float


def retrieve_historical_chunks(
    db: Session, query_text: str, top_k: int = 12, per_case_cap: int = 2
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks from the historical corpus."""
    import os
    from rank_bm25 import BM25Okapi
    
    top_k = int(os.getenv("TOP_K", str(top_k)))
    
    chunks = db.query(models.Chunk).filter(models.Chunk.historical_case_id.isnot(None)).all()
    if not chunks:
        return []

    provider = get_embedding_provider()
    query_vec = provider.embed([query_text])[0]
    tokenized_query = list(_keywords(query_text))

    tokenized_corpus = [list(_keywords(c.text)) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    lexical_scores = bm25.get_scores(tokenized_query)
    
    # Normalize lexical scores roughly
    max_lexical = max(lexical_scores) if lexical_scores.size > 0 and max(lexical_scores) > 0 else 1.0

    scored: List[RetrievedChunk] = []
    seen_texts = set()
    for i, c in enumerate(chunks):
        norm_text = " ".join(c.text.split())[:400]
        if norm_text in seen_texts:
            continue  # drop exact/near duplicate chunk text
        seen_texts.add(norm_text)

        sem = cosine_similarity(query_vec, c.embedding or [])
        kw = lexical_scores[i] / max_lexical
        hybrid = 0.7 * sem + 0.3 * kw
        scored.append(RetrievedChunk(chunk=c, semantic_score=sem, keyword_score=kw, hybrid_score=hybrid))

    scored.sort(key=lambda r: r.hybrid_score, reverse=True)

    # Diversify: cap chunks per historical case so one verbose case doesn't
    # dominate the result list, while still ranking by score overall.
    per_case_count: dict = {}
    diversified: List[RetrievedChunk] = []
    for r in scored:
        hc_id = r.chunk.historical_case_id
        if per_case_count.get(hc_id, 0) >= per_case_cap:
            continue
        diversified.append(r)
        per_case_count[hc_id] = per_case_count.get(hc_id, 0) + 1
        if len(diversified) >= top_k:
            break

    return diversified


def retrieve_case_document_chunks(
    db: Session, case_id: str, query_text: Optional[str] = None, top_k: int = 40
) -> List[models.Chunk]:
    """Return chunks belonging to the current case's own uploaded documents.

    Used for evidence/contradiction analysis where we scan the case's own
    material rather than the historical corpus.
    """
    chunks = db.query(models.Chunk).filter(models.Chunk.case_id == case_id).all()
    if query_text is None:
        return chunks[:top_k]

    provider = get_embedding_provider()
    query_vec = provider.embed([query_text])[0]
    scored = [(cosine_similarity(query_vec, c.embedding or []), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def aggregate_by_historical_case(retrieved: List[RetrievedChunk]) -> dict:
    """Group retrieved chunks by historical case, keeping the best score."""
    grouped: dict = {}
    for r in retrieved:
        hc_id = r.chunk.historical_case_id
        grouped.setdefault(hc_id, []).append(r)
    for hc_id in grouped:
        grouped[hc_id].sort(key=lambda r: r.hybrid_score, reverse=True)
    return grouped
