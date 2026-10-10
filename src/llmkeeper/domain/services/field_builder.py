from __future__ import annotations
import re
import numpy as np
from llmkeeper.domain.models import Anchor, Embedding, TopicField, Strictness

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?¿¡])\s+")
_MIN_WORDS = 3
_MAX_CHUNKS = 50


def split_into_chunks(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if _BULLET_RE.match(l)]
    if len(bullet_lines) >= 2:
        raw_chunks = [_BULLET_RE.sub("", l).strip() for l in bullet_lines]
    else:
        raw_chunks = _SENTENCE_SPLIT_RE.split(text.strip())

    chunks = []
    seen = set()
    for c in raw_chunks:
        c = c.strip()
        if len(c.split()) < _MIN_WORDS:
            continue
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        chunks.append(c)

    if len(chunks) > _MAX_CHUNKS:
        chunks = _merge_to_cap(chunks, _MAX_CHUNKS)
    return chunks or [text.strip()]


def _merge_to_cap(chunks: list[str], cap: int) -> list[str]:
    while len(chunks) > cap:
        merged = []
        for i in range(0, len(chunks) - 1, 2):
            merged.append(chunks[i] + " " + chunks[i + 1])
        if len(chunks) % 2:
            merged.append(chunks[-1])
        chunks = merged
    return chunks


def build_centroid(embeddings: list[Embedding]) -> Embedding:
    stacked = np.stack([e.vector for e in embeddings])
    mean = stacked.mean(axis=0)
    return Embedding(vector=mean.astype(np.float32), dim=mean.shape[0])


def compute_dispersion(embeddings: list[Embedding], centroid: Embedding) -> float:
    dists = [1.0 - e.cosine_similarity(centroid) for e in embeddings]
    return float(np.mean(dists)) if dists else 0.0


def build_topic_field(
    field_id: str,
    system_prompt: str,
    chunk_texts: list[str],
    chunk_embeddings: list[Embedding],
    extra_topic_texts: list[str] | None,
    extra_topic_embeddings: list[Embedding] | None,
    strictness: Strictness,
    alpha: float = 0.7,
    top_k: int = 3,
) -> TopicField:
    anchors = [Anchor(text=t, embedding=e) for t, e in zip(chunk_texts, chunk_embeddings)]
    if extra_topic_texts:
        anchors += [
            Anchor(text=t, embedding=e)
            for t, e in zip(extra_topic_texts, extra_topic_embeddings or [])
        ]
    centroid = build_centroid([a.embedding for a in anchors])
    dispersion = compute_dispersion([a.embedding for a in anchors], centroid)
    return TopicField(
        field_id=field_id,
        system_prompt=system_prompt,
        anchors=anchors,
        centroid=centroid,
        dispersion=dispersion,
        strictness=strictness,
        alpha=alpha,
        top_k=top_k,
    )
