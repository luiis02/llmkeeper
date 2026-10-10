import pytest
from llmkeeper.domain.models import StrictnessLevel, Embedding
from llmkeeper.application.use_cases import (
    CreateField,
    ValidateMessage,
    AdjustStrictness,
    CreateSession,
    AppendMessageToSession,
    GetSession,
    FieldNotFoundError,
    SessionNotFoundError,
)
from llmkeeper.adapters.outbound.repository_memory import (
    InMemoryFieldRepository,
    InMemorySessionRepository,
)
from tests.fakes import ScriptedEmbeddingProvider


def test_create_field():
    provider = ScriptedEmbeddingProvider(
        {
            "This is the first chunk.": [1.0, 0.0, 0.0],
            "This is the second chunk.": [0.0, 1.0, 0.0],
        }
    )
    field_repo = InMemoryFieldRepository()
    uc = CreateField(embeddings=provider, fields=field_repo)

    field = uc.execute(
        system_prompt="This is the first chunk. This is the second chunk.",
        strictness_preset=StrictnessLevel.ESTRICTO,
    )

    assert field.field_id is not None
    assert len(field.anchors) >= 2
    assert field.strictness.threshold == 0.65


def test_validate_message():
    provider = ScriptedEmbeddingProvider(
        {
            "This is the first chunk.": [1.0, 0.0],
            "This is on topic message": [0.95, 0.05],
            "This is off topic completely": [0.0, 1.0],
        }
    )
    field_repo = InMemoryFieldRepository()
    create_uc = CreateField(embeddings=provider, fields=field_repo)
    field = create_uc.execute(
        system_prompt="This is the first chunk.",
        strictness_preset=StrictnessLevel.MODERADO,
    )

    validate_uc = ValidateMessage(embeddings=provider, fields=field_repo)

    result_on = validate_uc.execute(field.field_id, "This is on topic message")
    assert result_on.is_valid is True

    result_off = validate_uc.execute(field.field_id, "This is off topic completely")
    assert result_off.is_valid is False


def test_validate_message_field_not_found():
    provider = ScriptedEmbeddingProvider({})
    field_repo = InMemoryFieldRepository()
    validate_uc = ValidateMessage(embeddings=provider, fields=field_repo)

    with pytest.raises(FieldNotFoundError):
        validate_uc.execute("nonexistent", "msg")


def test_adjust_strictness():
    provider = ScriptedEmbeddingProvider({"This is the first chunk.": [1.0, 0.0]})
    field_repo = InMemoryFieldRepository()
    create_uc = CreateField(embeddings=provider, fields=field_repo)
    field = create_uc.execute(
        system_prompt="This is the first chunk.",
        strictness_preset=StrictnessLevel.MODERADO,
    )

    adjust_uc = AdjustStrictness(fields=field_repo)
    adjust_uc.execute(field.field_id, preset=StrictnessLevel.MUY_ESTRICTO)

    updated = field_repo.get(field.field_id)
    assert updated.strictness.threshold == 0.75


def test_create_session():
    provider = ScriptedEmbeddingProvider({"This is the first chunk.": [1.0, .0]})
    field_repo = InMemoryFieldRepository()
    session_repo = InMemorySessionRepository()

    create_field_uc = CreateField(embeddings=provider, fields=field_repo)
    field = create_field_uc.execute(system_prompt="This is the first chunk.")

    create_session_uc = CreateSession(fields=field_repo, sessions=session_repo)
    session = create_session_uc.execute(
        field.field_id, patience=5, rolling_window=3
    )

    assert session.session_id is not None
    assert session.field_id == field.field_id
    assert session.patience == 5


def test_append_message_to_session():
    provider = ScriptedEmbeddingProvider(
        {
            "This is the first chunk.": [1.0, 0.0],
            "This is on topic message": [0.9, 0.1],
            "This is off topic completely": [0.0, 1.0],
        }
    )
    field_repo = InMemoryFieldRepository()
    session_repo = InMemorySessionRepository()

    create_field_uc = CreateField(embeddings=provider, fields=field_repo)
    field = create_field_uc.execute(system_prompt="This is the first chunk.")

    create_session_uc = CreateSession(fields=field_repo, sessions=session_repo)
    session = create_session_uc.execute(field.field_id, patience=2)

    append_uc = AppendMessageToSession(
        embeddings=provider, fields=field_repo, sessions=session_repo
    )

    session = append_uc.execute(session.session_id, "This is on topic message")
    assert len(session.history) == 1
    assert session.consecutive_off == 0

    session = append_uc.execute(session.session_id, "This is off topic completely")
    assert len(session.history) == 2
    assert session.consecutive_off == 1


def test_get_session():
    provider = ScriptedEmbeddingProvider({"This is the first chunk.": [1.0, 0.0]})
    field_repo = InMemoryFieldRepository()
    session_repo = InMemorySessionRepository()

    create_field_uc = CreateField(embeddings=provider, fields=field_repo)
    field = create_field_uc.execute(system_prompt="This is the first chunk.")

    create_session_uc = CreateSession(fields=field_repo, sessions=session_repo)
    session = create_session_uc.execute(field.field_id)

    get_uc = GetSession(sessions=session_repo)
    retrieved = get_uc.execute(session.session_id)

    assert retrieved.session_id == session.session_id


def test_get_session_not_found():
    session_repo = InMemorySessionRepository()
    get_uc = GetSession(sessions=session_repo)

    with pytest.raises(SessionNotFoundError):
        get_uc.execute("nonexistent")
