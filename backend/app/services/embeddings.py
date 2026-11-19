import os
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]: ...

class LocalHashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, n_features: int = EMBEDDING_DIM):
        self._vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._vectorizer.transform(texts).toarray().astype(float).tolist() if texts else []

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        try: from sentence_transformers import SentenceTransformer
        except ImportError: raise RuntimeError("sentence-transformers not installed.")
        self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).astype(float).tolist() if texts else []

_provider_singleton: EmbeddingProvider | None = None
_provider_singleton_name: str | None = None

def get_embedding_provider() -> EmbeddingProvider:
    global _provider_singleton, _provider_singleton_name
    name = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    dim = int(os.getenv("EMBEDDING_DIM", "512"))
    key = f"{name}_{dim}"
    if _provider_singleton is None or _provider_singleton_name != key:
        if name == "local": _provider_singleton = LocalHashingEmbeddingProvider(n_features=dim)
        elif name == "sentence-transformers": _provider_singleton = SentenceTransformerEmbeddingProvider()
        else: raise ValueError(f"Unknown EMBEDDING_PROVIDER '{name}'")
        _provider_singleton_name = key
    return _provider_singleton

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(np.dot(va, vb) / (na * nb)) if (na > 0 and nb > 0) else 0.0
