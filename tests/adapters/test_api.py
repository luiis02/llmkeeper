import pytest
from fastapi.testclient import TestClient
from llmkeeper.infrastructure.config import Settings
from llmkeeper.infrastructure.container import Container
from llmkeeper.adapters.outbound.repository_memory import (
    InMemoryFieldRepository,
    InMemorySessionRepository,
)
from llmkeeper.adapters.inbound.api import create_router
from tests.fakes import ScriptedEmbeddingProvider
from fastapi import FastAPI


@pytest.fixture
def test_container():
    provider = ScriptedEmbeddingProvider(
        {
            "This is the first chunk.": [1.0, 0.0, 0.0],
            "This is the second chunk.": [0.0, 1.0, 0.0],
            "This is on topic message": [0.95, 0.05, 0.0],
            "This is off topic completely": [0.0, 0.0, 1.0],
        }
    )
    settings = Settings()
    container = Container(settings=settings, embedding_provider=provider)
    return container


@pytest.fixture
def client(test_container):
    app = FastAPI()
    router = create_router(
        create_field_uc=test_container.create_field_uc(),
        validate_message_uc=test_container.validate_message_uc(),
        adjust_strictness_uc=test_container.adjust_strictness_uc(),
        create_session_uc=test_container.create_session_uc(),
        append_message_uc=test_container.append_message_uc(),
        get_session_uc=test_container.get_session_uc(),
    )
    app.include_router(router)
    return TestClient(app)


def test_post_fields(client):
    response = client.post(
        "/api/v1/fields",
        json={
            "system_prompt": "This is the first chunk. This is the second chunk.",
            "strictness_preset": "moderado",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "field_id" in data
    assert data["num_anchors"] >= 2


def test_get_fields(client):
    create_resp = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "estricto"},
    )
    field_id = create_resp.json()["field_id"]

    get_resp = client.get(f"/api/v1/fields/{field_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["field_id"] == field_id


def test_patch_strictness(client):
    create_resp = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_resp.json()["field_id"]

    patch_resp = client.patch(
        f"/api/v1/fields/{field_id}/strictness",
        json={"preset": "muy_estricto"},
    )
    assert patch_resp.status_code == 200

    get_resp = client.get(f"/api/v1/fields/{field_id}")
    assert abs(get_resp.json()["threshold"] - 0.75) < 0.001


def test_validate_message_on_topic(client):
    create_resp = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_resp.json()["field_id"]

    validate_resp = client.post(
        f"/api/v1/fields/{field_id}/validate",
        json={"text": "This is on topic message"},
    )
    assert validate_resp.status_code == 200
    assert validate_resp.json()["is_valid"] is True


def test_validate_message_off_topic(client):
    create_resp = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_resp.json()["field_id"]

    validate_resp = client.post(
        f"/api/v1/fields/{field_id}/validate",
        json={"text": "This is off topic completely"},
    )
    assert validate_resp.status_code == 200
    assert validate_resp.json()["is_valid"] is False


def test_create_session(client):
    create_field = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_field.json()["field_id"]

    create_session_resp = client.post(
        f"/api/v1/fields/{field_id}/sessions",
        json={"patience": 3, "rolling_window": 5},
    )
    assert create_session_resp.status_code == 200
    data = create_session_resp.json()
    assert "session_id" in data
    assert data["status"] == "on_topic"


def test_append_message_to_session(client):
    create_field = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_field.json()["field_id"]

    create_session_resp = client.post(
        f"/api/v1/fields/{field_id}/sessions",
        json={"patience": 2, "rolling_window": 5},
    )
    session_id = create_session_resp.json()["session_id"]

    msg_resp = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"text": "This is on topic message"},
    )
    assert msg_resp.status_code == 200
    assert msg_resp.json()["is_valid"] is True
    assert msg_resp.json()["status"] == "on_topic"


def test_session_warning_and_off_topic(client):
    create_field = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_field.json()["field_id"]

    create_session_resp = client.post(
        f"/api/v1/fields/{field_id}/sessions",
        json={"patience": 3, "rolling_window": 5},
    )
    session_id = create_session_resp.json()["session_id"]

    for i in range(4):
        msg_resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"text": "This is off topic completely"},
        )
        if i < 2:
            assert msg_resp.json()["status"] == "warning"
        else:
            assert msg_resp.json()["status"] == "off_topic"


def test_get_session(client):
    create_field = client.post(
        "/api/v1/fields",
        json={"system_prompt": "This is the first chunk.", "strictness_preset": "moderado"},
    )
    field_id = create_field.json()["field_id"]

    create_session_resp = client.post(
        f"/api/v1/fields/{field_id}/sessions",
        json={"patience": 3, "rolling_window": 5},
    )
    session_id = create_session_resp.json()["session_id"]

    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id
