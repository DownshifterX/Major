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


def _ensure_corpus_cache(db: Session, force_reload: bool = False):
    global _CORPUS_CACHE
    if _CORPUS_CACHE["loaded"] and not force_reload:
        return

    import os
    import pickle
    from pathlib import Path
    from ..database import DATA_DIR
    cache_file = Path(DATA_DIR) / "corpus_cache.pkl"

    # In testing mode or when explicitly disabled, do not read disk cache
    use_disk_cache = os.getenv("USE_DISK_CORPUS_CACHE", "1") == "1"

    if use_disk_cache and cache_file.exists() and not force_reload:
        try:
            with open(cache_file, "rb") as f:
                loaded_data = pickle.load(f)
            _CORPUS_CACHE = loaded_data
            _CORPUS_CACHE["loaded"] = True
            return
        except Exception:
            pass

    # Load from database if cache file does not exist
    chunks = (
        db.query(
            models.Chunk.id,
            models.Chunk.historical_case_id,
            models.Chunk.text,
            models.Chunk.embedding,
        )
        .filter(models.Chunk.historical_case_id.isnot(None))
        .all()
    )

    if not chunks:
        _CORPUS_CACHE["loaded"] = True
        return

    from rank_bm25 import BM25Plus

    tokenized_corpus = []
    chunk_objs = []
    norm_texts = []
    valid_embeddings = []

    for c in chunks:
        c_obj = models.Chunk(
            id=c.id,
            historical_case_id=c.historical_case_id,
            text=c.text,
            embedding=c.embedding,
        )
        chunk_objs.append(c_obj)
        tokenized_corpus.append(list(_keywords(c.text or "")))
        norm_texts.append(" ".join((c.text or "").split())[:400])
        valid_embeddings.append(c.embedding if c.embedding else [0.0] * 384)

    bm25 = BM25Plus(tokenized_corpus)
    emb_arr = np.array(valid_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix_normed = emb_arr / norms

    _CORPUS_CACHE = {
        "loaded": True,
        "chunks": chunk_objs,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
        "embedding_matrix": emb_matrix_normed,
        "norm_texts": norm_texts,
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(_CORPUS_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def retrieve_historical_chunks(
    db: Session, query_text: str, top_k: int = 12, per_case_cap: int = 2
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks from the historical corpus."""
    import os

    top_k = int(os.getenv("TOP_K", str(top_k)))

    _ensure_corpus_cache(db)
    chunks = _CORPUS_CACHE["chunks"]
    if not chunks:
        return []

    provider = get_embedding_provider()
    query_vec = provider.embed([query_text])[0]
    tokenized_query = list(_keywords(query_text))

    # Fast BM25 score retrieval
    bm25 = _CORPUS_CACHE["bm25"]
    lexical_scores = bm25.get_scores(tokenized_query)
    clamped_scores = [max(0.0, float(s)) for s in lexical_scores]
    max_lexical = max(clamped_scores) if clamped_scores and max(clamped_scores) > 0 else 1.0

    # Fast vectorized semantic similarity: 1 matrix-vector multiplication
    q_vec = np.array(query_vec, dtype=np.float32)
    mat = _CORPUS_CACHE["embedding_matrix"]
    if q_vec.ndim == 1 and mat is not None and mat.shape[1] != q_vec.shape[0]:
        # Dimension mismatch between chunk embedding and query provider (e.g. test fixture vs live model)
        if q_vec.shape[0] < mat.shape[1]:
            q_vec = np.pad(q_vec, (0, mat.shape[1] - q_vec.shape[0]))
        else:
            q_vec = q_vec[:mat.shape[1]]

    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0 and mat is not None and len(mat) > 0:
        q_vec = q_vec / q_norm
        sem_scores = np.dot(mat, q_vec)
    else:
        sem_scores = np.zeros(len(chunks), dtype=np.float32)

    lex_scores_arr = np.array(clamped_scores, dtype=np.float32) / float(max_lexical)
    hybrid_scores_arr = 0.7 * sem_scores + 0.3 * lex_scores_arr

    # Pre-select top candidate indices (e.g. top 500) to avoid looping over 110,000 objects in Python
    candidate_indices = np.argpartition(hybrid_scores_arr, -min(500, len(chunks)))[-min(500, len(chunks)):]
    candidate_indices = candidate_indices[np.argsort(-hybrid_scores_arr[candidate_indices])]

    norm_texts = _CORPUS_CACHE["norm_texts"]
    seen_texts = set()
    per_case_count: dict = {}
    diversified: List[RetrievedChunk] = []

    for i in candidate_indices:
        nt = norm_texts[i]
        if nt in seen_texts:
            continue
        seen_texts.add(nt)

        c = chunks[i]
        hc_id = c.historical_case_id
        if per_case_count.get(hc_id, 0) >= per_case_cap:
            continue

        sem = float(sem_scores[i])
        kw = float(lex_scores_arr[i])
        hybrid = float(hybrid_scores_arr[i])

        diversified.append(
            RetrievedChunk(
                chunk=c,
                semantic_score=sem,
                keyword_score=kw,
                hybrid_score=hybrid,
            )
        )
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
