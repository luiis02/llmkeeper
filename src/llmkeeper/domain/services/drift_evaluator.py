from __future__ import annotations
from llmkeeper.domain.models import (
    Embedding,
    TopicField,
    ValidationResult,
    Session,
    MessageScore,
    SessionStatus,
)


def score_against_field(field: TopicField, message_embedding: Embedding) -> ValidationResult:
    sims = [(a.text, a.embedding.cosine_similarity(message_embedding)) for a in field.anchors]
    sims.sort(key=lambda p: p[1], reverse=True)
    top = sims[: field.top_k] or [("", 0.0)]
    top_avg = sum(s for _, s in top) / len(top)
    centroid_sim = field.centroid.cosine_similarity(message_embedding)
    score = field.alpha * top_avg + (1 - field.alpha) * centroid_sim
    threshold = field.strictness.threshold
    return ValidationResult(
        is_valid=score >= threshold,
        score=score,
        threshold=threshold,
        nearest_anchor_text=top[0][0],
    )


def append_message_to_session(
    session: Session, text: str, result: ValidationResult
) -> Session:
    session.history.append(
        MessageScore(
            text=text,
            score=result.score,
            is_valid=result.is_valid,
            threshold=result.threshold,
        )
    )
    if result.is_valid:
        session.consecutive_off = 0
    else:
        session.consecutive_off += 1

    if session.consecutive_off == 0:
        session.status = SessionStatus.ON_TOPIC
    elif session.consecutive_off < session.patience:
        session.status = SessionStatus.WARNING
    else:
        session.status = SessionStatus.OFF_TOPIC
    return session
