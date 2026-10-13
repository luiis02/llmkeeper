import pytest
from llmkeeper.domain.models import Embedding, Strictness, StrictnessLevel
from llmkeeper.domain.services.field_builder import (
    split_into_chunks,
    build_centroid,
    compute_dispersion,
    build_topic_field,
)


def test_split_into_chunks_with_bullets():
    text = """
    - First point about the topic
    - Second point about the topic
    - Third point about the topic
    """
    chunks = split_into_chunks(text)
    assert len(chunks) == 3
    assert "First point" in chunks[0]


def test_split_into_chunks_with_sentences():
    text = (
        "This is the first sentence. This is the second sentence. This is the third."
    )
    chunks = split_into_chunks(text)
    assert len(chunks) == 3


def test_split_into_chunks_filters_short():
    text = "Short. This is a proper sentence. OK."
    chunks = split_into_chunks(text)
    assert len(chunks) == 1
    assert "proper" in chunks[0]


def test_split_into_chunks_deduplicates():
    text = "Same sentence here. Same sentence here."
    chunks = split_into_chunks(text)
    assert len(chunks) == 1


def test_build_centroid():
    emb1 = Embedding.from_list([1.0, 0.0, 0.0, 0.0])
    emb2 = Embedding.from_list([0.0, 1.0, 0.0, 0.0])
    emb3 = Embedding.from_list([0.0, 0.0, 1.0, 0.0])

    centroid = build_centroid([emb1, emb2, emb3])
    assert centroid.dim == 4
    expected = [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0]
    for i, val in enumerate(expected):
        assert abs(centroid.vector[i] - val) < 0.001


def test_compute_dispersion():
    emb1 = Embedding.from_list([1.0, 0.0])
    emb2 = Embedding.from_list([0.0, 1.0])
    centroid = Embedding.from_list([0.5, 0.5])

    disp = compute_dispersion([emb1, emb2], centroid)
    assert 0.0 <= disp <= 1.0


def test_build_topic_field():
    chunk_texts = ["first chunk", "second chunk"]
    chunk_embeddings = [
        Embedding.from_list([1.0, 0.0, 0.0]),
        Embedding.from_list([0.0, 1.0, 0.0]),
    ]
    strictness = Strictness.from_preset(StrictnessLevel.MODERADO)

    field = build_topic_field(
        field_id="test_field",
        system_prompt="Test prompt",
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        extra_topic_texts=None,
        extra_topic_embeddings=None,
        strictness=strictness,
    )

    assert field.field_id == "test_field"
    assert len(field.anchors) == 2
    assert field.centroid.dim == 3
    assert field.strictness.threshold == 0.22
