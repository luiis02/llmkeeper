from sentence_transformers import SentenceTransformer
from llmkeeper.domain.models import Embedding


class SentenceTransformerEmbeddingAdapter:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> Embedding:
        vec = self._model.encode(text, normalize_embeddings=False)
        return Embedding.from_list(vec.tolist())

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        vecs = self._model.encode(texts, normalize_embeddings=False, batch_size=32)
        return [Embedding.from_list(v.tolist()) for v in vecs]
