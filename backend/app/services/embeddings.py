"""Embedding provider abstraction.

EmbeddingProvider.embed(texts) -> list[list[float]]

Default provider (`local`) uses a deterministic hashing-based TF-style
vectorizer (scikit-learn's HashingVectorizer). It needs no network call and
no downloaded model, which keeps the PoC fully self-contained, but it is
still a genuine text vectorization (not a random/fake vector) — near-duplicate
and topically-similar text produce cosine-similar vectors. This is clearly a
lexical/statistical embedding, not a deep semantic embedding.

Swap in a real semantic embedding API (OpenAI, Voyage, Cohere, an in-house
model, etc.) by implementing another EmbeddingProvider and changing
EMBEDDING_PROVIDER in the environment — nothing else in the pipeline needs to
change because retrieval only depends on this interface.
"""
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding for offline demo use."""

    def __init__(self, n_features: int = EMBEDDING_DIM):
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real semantic embedding using BERT-family models (sentence-transformers)."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not installed.")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(float).tolist()


_provider_singleton: EmbeddingProvider | None = None
_provider_singleton_name: str | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider_singleton, _provider_singleton_name
    name = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    dim = int(os.getenv("EMBEDDING_DIM", "512"))
    key = f"{name}_{dim}"
    if _provider_singleton is None or _provider_singleton_name != key:
        if name == "local":
            _provider_singleton = LocalHashingEmbeddingProvider(n_features=dim)
        elif name == "sentence-transformers":
            _provider_singleton = SentenceTransformerEmbeddingProvider()
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER '{name}'")
        _provider_singleton_name = key
    return _provider_singleton


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0  # dimension mismatch or empty vectors
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
