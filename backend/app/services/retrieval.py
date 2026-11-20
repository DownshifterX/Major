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


import numpy as np

# In-memory corpus cache for ultra-fast RAG retrieval across queries
_CORPUS_CACHE = {
    "loaded": False,
    "chunks": [],
    "tokenized_corpus": [],
    "bm25": None,
    "embedding_matrix": None,  # (N, D) normalized numpy array
    "norm_texts": [],
}


def retrieve_historical_chunks(
    db: Session, query_text: str, top_k: int = 12, per_case_cap: int = 2
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks from the historical corpus using SQLite FTS5 + Semantic embeddings."""
    import os

    top_k = int(os.getenv("TOP_K", str(top_k)))

    # 1. Quick Keyword Extraction for SQLite FTS5 search
    keywords = [w for w in _keywords(query_text) if len(w) > 3 and not w.isdigit()]
    fts_query = " OR ".join(keywords[:15]) if keywords else "land"

    # Search FTS5 virtual table directly to get top candidates quickly (sub-10ms)
    conn = db.connection().connection
    try:
        sql = """
            SELECT c.id, c.historical_case_id, c.text, c.embedding, f.rank
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ? AND c.historical_case_id IS NOT NULL
            ORDER BY f.rank
            LIMIT ?
        """
        candidate_rows = conn.execute(sql, (fts_query, max(top_k * 10, 60))).fetchall()
    except Exception:
        candidate_rows = []

    # If FTS returns empty (e.g. no match), fallback to sample historical chunks
    if not candidate_rows:
        try:
            sql_fallback = """
                SELECT id, historical_case_id, text, embedding, 0.0
                FROM chunks
                WHERE historical_case_id IS NOT NULL
                LIMIT ?
            """
            candidate_rows = conn.execute(sql_fallback, (max(top_k * 5, 20),)).fetchall()
        except Exception:
            return []

    # 2. Score candidates semantically with embedding provider
    provider = get_embedding_provider()
    try:
        query_vec = provider.embed([query_text[:500]])[0]
    except Exception:
        query_vec = [0.0] * 384

    query_kw = set(keywords)
    scored_candidates = []
    seen_texts = set()
    per_case_count: dict = {}

    for row in candidate_rows:
        cid, hc_id, text, emb, fts_rank = row[0], row[1], row[2] or "", row[3], float(row[4])
        norm_txt = " ".join(text.split())[:300]
        if norm_txt in seen_texts:
            continue
        seen_texts.add(norm_txt)

        # Lexical score from keyword overlap or FTS rank
        kw_score = _keyword_overlap(query_kw, _keywords(text))
        
        # Semantic score
        import json
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = []
        sem_score = cosine_similarity(query_vec, emb) if emb else kw_score

        # Hybrid score combines semantic and lexical relevance
        hybrid_score = 0.65 * sem_score + 0.35 * kw_score

        chunk_obj = models.Chunk(id=cid, historical_case_id=hc_id, text=text, embedding=emb)
        scored_candidates.append(
            RetrievedChunk(
                chunk=chunk_obj,
                semantic_score=sem_score,
                keyword_score=kw_score,
                hybrid_score=hybrid_score,
            )
        )

    # Sort candidates by hybrid score descending
    scored_candidates.sort(key=lambda x: x.hybrid_score, reverse=True)

    # Apply per-case cap and top_k
    diversified: List[RetrievedChunk] = []
    for sc in scored_candidates:
        hc_id = sc.chunk.historical_case_id
        if per_case_count.get(hc_id, 0) >= per_case_cap:
            continue
        diversified.append(sc)
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
