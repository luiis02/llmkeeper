from __future__ import annotations
from llmkeeper.domain.models import TopicField, Session, Strictness, StrictnessLevel


class InMemoryFieldRepository:
    def __init__(self) -> None:
        self._store: dict[str, TopicField] = {}

    def save(self, field: TopicField) -> None:
        self._store[field.field_id] = field

    def get(self, field_id: str) -> TopicField | None:
        return self._store.get(field_id)

    def update_strictness(
        self, field_id: str, new_threshold: float, preset_name: str | None
    ) -> None:
        if field_id not in self._store:
            return
        field = self._store[field_id]
        field.strictness = Strictness(
            threshold=new_threshold,
            preset=StrictnessLevel(preset_name) if preset_name else None,
        )


class InMemorySessionRepository:
    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self._store[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        return self._store.get(session_id)
