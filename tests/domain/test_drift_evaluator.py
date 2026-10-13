import pytest
from llmkeeper.domain.models import (
    Embedding,
    Anchor,
    TopicField,
    Strictness,
    StrictnessLevel,
    Session,
    SessionStatus,
)
from llmkeeper.domain.services.drift_evaluator import (
    score_against_field,
    append_message_to_session,
)


def test_score_against_field_similar():
    anchor1 = Anchor(text="anchor 1", embedding=Embedding.from_list([1.0, 0.0, 0.0]))
    anchor2 = Anchor(text="anchor 2", embedding=Embedding.from_list([0.0, 1.0, 0.0]))
    centroid = Embedding.from_list([0.5, 0.5, 0.0])

    field = TopicField(
        field_id="test",
        system_prompt="test",
        anchors=[anchor1, anchor2],
        centroid=centroid,
        dispersion=0.5,
        strictness=Strictness.from_preset(StrictnessLevel.MODERADO),
        alpha=0.7,
        top_k=2,
    )

    msg_emb = Embedding.from_list([0.95, 0.05, 0.0])
    result = score_against_field(field, msg_emb)

    assert result.is_valid is True
    assert result.score > 0.55
    assert result.threshold == 0.22


def test_score_against_field_dissimilar():
    anchor1 = Anchor(text="anchor 1", embedding=Embedding.from_list([1.0, 0.0, 0.0]))
    anchor2 = Anchor(text="anchor 2", embedding=Embedding.from_list([0.0, 1.0, 0.0]))
    centroid = Embedding.from_list([0.5, 0.5, 0.0])

    field = TopicField(
        field_id="test",
        system_prompt="test",
        anchors=[anchor1, anchor2],
        centroid=centroid,
        dispersion=0.5,
        strictness=Strictness.from_preset(StrictnessLevel.MODERADO),
        alpha=0.7,
        top_k=2,
    )

    msg_emb = Embedding.from_list([0.0, 0.0, 1.0])
    result = score_against_field(field, msg_emb)

    assert result.is_valid is False
    assert result.score < 0.55


def test_append_message_on_topic():
    session = Session(
        session_id="test",
        field_id="field",
        patience=3,
        rolling_window=5,
    )

    from llmkeeper.domain.models import ValidationResult

    result = ValidationResult(
        is_valid=True, score=0.8, threshold=0.55, nearest_anchor_text="anchor"
    )

    session = append_message_to_session(session, "test message", result)

    assert len(session.history) == 1
    assert session.history[0].is_valid is True
    assert session.consecutive_off == 0
    assert session.status == SessionStatus.ON_TOPIC


def test_append_message_warning():
    session = Session(
        session_id="test",
        field_id="field",
        patience=3,
        rolling_window=5,
    )

    from llmkeeper.domain.models import ValidationResult

    result = ValidationResult(
        is_valid=False, score=0.3, threshold=0.55, nearest_anchor_text="anchor"
    )

    session = append_message_to_session(session, "off-topic 1", result)
    assert session.status == SessionStatus.WARNING
    assert session.consecutive_off == 1

    session = append_message_to_session(session, "off-topic 2", result)
    assert session.status == SessionStatus.WARNING
    assert session.consecutive_off == 2


def test_append_message_off_topic():
    session = Session(
        session_id="test",
        field_id="field",
        patience=2,
        rolling_window=5,
    )

    from llmkeeper.domain.models import ValidationResult

    result = ValidationResult(
        is_valid=False, score=0.3, threshold=0.55, nearest_anchor_text="anchor"
    )

    for i in range(3):
        session = append_message_to_session(session, f"off-topic {i}", result)

    assert session.status == SessionStatus.OFF_TOPIC
    assert session.consecutive_off == 3


def test_rolling_average():
    session = Session(
        session_id="test",
        field_id="field",
        patience=3,
        rolling_window=2,
    )

    from llmkeeper.domain.models import ValidationResult

    scores = [0.8, 0.7, 0.6]
    for score in scores:
        result = ValidationResult(
            is_valid=score >= 0.55,
            score=score,
            threshold=0.55,
            nearest_anchor_text="anchor",
        )
        session = append_message_to_session(session, "msg", result)

    avg = session.rolling_average()
    expected = (0.7 + 0.6) / 2
    assert abs(avg - expected) < 0.01
