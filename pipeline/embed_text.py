"""Step 4a: text embeddings. bge-m3 dense (1024-d, normalised) and BM25 sparse vectors, for chunks and queries."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from pipeline.config import settings
from pipeline.parsing import resolve_device


class TextEmbedder:
    def __init__(self) -> None:
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        s = settings()
        self.model = SentenceTransformer(s.text_model, device=resolve_device())
        self.model.max_seq_length = 1024
        self.bm25 = SparseTextEmbedding(s.bm25_model)

    def dense(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def sparse(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in self.bm25.embed(texts, batch_size=256)] if texts else []

    def dense_query(self, q: str) -> np.ndarray:
        return self.dense([q])[0]

    def sparse_query(self, q: str) -> tuple[list[int], list[float]]:
        e = next(self.bm25.query_embed(q))
        return e.indices.tolist(), e.values.tolist()


@lru_cache
def text_embedder() -> TextEmbedder:
    return TextEmbedder()
