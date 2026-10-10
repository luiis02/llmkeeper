import pytest


@pytest.mark.integration
def test_full_flow_with_real_embeddings():
    """Integration test with real SentenceTransformer model."""
    from fastapi.testclient import TestClient
    from llmkeeper.infrastructure.app import create_app
    from llmkeeper.infrastructure.config import Settings

    settings = Settings()
    app = create_app(settings)
    client = TestClient(app)

    create_field_resp = client.post(
        "/api/v1/fields",
        json={
            "system_prompt": "Eres un experto en recetas de cocina. Ayuda a los usuarios con ingredientes y técnicas de cocina.",
            "strictness_preset": "moderado",
        },
    )
    assert create_field_resp.status_code == 200
    field_id = create_field_resp.json()["field_id"]

    validate_on_topic = client.post(
        f"/api/v1/fields/{field_id}/validate",
        json={"text": "¿Cuáles son los mejores ingredientes para hacer una paella?"},
    )
    assert validate_on_topic.status_code == 200
    assert validate_on_topic.json()["is_valid"] is True

    validate_off_topic = client.post(
        f"/api/v1/fields/{field_id}/validate",
        json={
            "text": "¿Cuál es la capital de Francia y cuáles son sus principales monumentos?"
        },
    )
    assert validate_off_topic.status_code == 200
    assert validate_off_topic.json()["is_valid"] is False
