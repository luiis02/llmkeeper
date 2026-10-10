from __future__ import annotations
from dataclasses import dataclass
from llmkeeper.domain.models import (
    TopicField,
    Strictness,
    StrictnessLevel,
    Session,
    ValidationResult,
)
from llmkeeper.domain.services.field_builder import split_into_chunks, build_topic_field
from llmkeeper.domain.services.drift_evaluator import (
    score_against_field,
    append_message_to_session,
)
from llmkeeper.ports.outbound import (
    EmbeddingProviderPort,
    FieldRepositoryPort,
    SessionRepositoryPort,
)


class FieldNotFoundError(Exception):
    pass


class SessionNotFoundError(Exception):
    pass


@dataclass
class CreateField:
    embeddings: EmbeddingProviderPort
    fields: FieldRepositoryPort

    def execute(
        self,
        system_prompt: str,
        extra_topics: list[str] | None = None,
        strictness_preset: StrictnessLevel | None = None,
        strictness_value: float | None = None,
    ) -> TopicField:
        strictness = _resolve_strictness(strictness_preset, strictness_value)
        chunk_texts = split_into_chunks(system_prompt)
        chunk_embeddings = self.embeddings.embed_batch(chunk_texts)
        extra_embeddings = (
            self.embeddings.embed_batch(extra_topics) if extra_topics else None
        )

        topic_field = build_topic_field(
            field_id=TopicField.new_id(),
            system_prompt=system_prompt,
            chunk_texts=chunk_texts,
            chunk_embeddings=chunk_embeddings,
            extra_topic_texts=extra_topics,
            extra_topic_embeddings=extra_embeddings,
            strictness=strictness,
        )
        self.fields.save(topic_field)
        return topic_field


@dataclass
class ValidateMessage:
    embeddings: EmbeddingProviderPort
    fields: FieldRepositoryPort

    def execute(self, field_id: str, text: str) -> ValidationResult:
        topic_field = self.fields.get(field_id)
        if topic_field is None:
            raise FieldNotFoundError(field_id)
        embedding = self.embeddings.embed(text)
        return score_against_field(topic_field, embedding)


@dataclass
class AdjustStrictness:
    fields: FieldRepositoryPort

    def execute(
        self,
        field_id: str,
        preset: StrictnessLevel | None = None,
        custom_value: float | None = None,
    ) -> None:
        if self.fields.get(field_id) is None:
            raise FieldNotFoundError(field_id)
        strictness = _resolve_strictness(preset, custom_value)
        self.fields.update_strictness(
            field_id, strictness.threshold, preset.value if preset else None
        )


@dataclass
class CreateSession:
    fields: FieldRepositoryPort
    sessions: SessionRepositoryPort

    def execute(
        self, field_id: str, patience: int = 3, rolling_window: int = 5
    ) -> Session:
        if self.fields.get(field_id) is None:
            raise FieldNotFoundError(field_id)
        session = Session(
            session_id=Session.new_id(),
            field_id=field_id,
            patience=patience,
            rolling_window=rolling_window,
        )
        self.sessions.save(session)
        return session


@dataclass
class AppendMessageToSession:
    embeddings: EmbeddingProviderPort
    fields: FieldRepositoryPort
    sessions: SessionRepositoryPort

    def execute(self, session_id: str, text: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        topic_field = self.fields.get(session.field_id)
        if topic_field is None:
            raise FieldNotFoundError(session.field_id)
        embedding = self.embeddings.embed(text)
        result = score_against_field(topic_field, embedding)
        session = append_message_to_session(session, text, result)
        self.sessions.save(session)
        return session


@dataclass
class GetSession:
    sessions: SessionRepositoryPort

    def execute(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session


def _resolve_strictness(
    preset: StrictnessLevel | None, custom_value: float | None
) -> Strictness:
    if custom_value is not None:
        return Strictness.from_custom(custom_value)
    return Strictness.from_preset(preset or StrictnessLevel.MODERADO)
