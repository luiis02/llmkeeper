import hashlib
import numpy as np
from llmkeeper.domain.models import Embedding


class FakeEmbeddingProvider:
    """Deterministic hash-based embedding, no ML model loaded."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, text: str) -> Embedding:
        h = hashlib.sha256(text.lower().encode()).digest()
        vec = np.frombuffer(h[: self.dim * 4], dtype=np.uint32).astype(np.float32)
        vec = (vec / vec.max()) if vec.max() else vec
        return Embedding(vector=vec, dim=self.dim)

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return [self.embed(t) for t in texts]


class ScriptedEmbeddingProvider:
    """Returns pre-defined vectors per text, for deterministic scoring tests."""

    def __init__(
        self, mapping: dict[str, list[float]], default_dim: int = 4
    ) -> None:
        self.mapping = mapping
        self.default_dim = default_dim

    def embed(self, text: str) -> Embedding:
        return Embedding.from_list(self.mapping.get(text, [0.0] * self.default_dim))

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return [self.embed(t) for t in texts]
