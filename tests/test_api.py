"""Integration and API endpoint tests for FinDecision.

Validates HTTP responses, status codes, schemas, and error handling using FastAPI TestClient
with mocked ML model inferences.
"""

from fastapi.testclient import TestClient

from app.main import app, model_service


class FakeConfig:
    """Mock configuration mirroring Hugging Face PretrainedConfig."""

    num_labels = 3
    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }


class FakeModel:
    """Mock sequence classification model."""

    config = FakeConfig()


class FakeTokenizer:
    """Mock tokenizer providing context window boundaries."""

    model_max_length = 128


def setup_fake_model(monkeypatch):
    """Fixture helper to monkeypatch model_service with lightweight deterministic stubs."""
    monkeypatch.setattr(
        model_service, "load", lambda: None
    )
    monkeypatch.setattr(
        model_service, "model", FakeModel()
    )
    monkeypatch.setattr(
        model_service, "tokenizer", FakeTokenizer()
    )
    monkeypatch.setattr(
        model_service,
        "predict_logits",
        lambda text: [-1.0, 0.0, 2.0],
    )


def test_health(monkeypatch):
    """Verify that /health returns 200 and reports model_loaded=True when initialized."""
    setup_fake_model(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_text_prediction(monkeypatch):
    """Verify that /predict/text parses payload, classifies sentiment, and returns normalized probabilities."""
    setup_fake_model(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/predict/text",
            json={
                "text": (
                    "The company reported stronger revenue."
                )
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] == "positive"
    assert result["input_type"] == "text"
    assert set(result["probabilities"]) == {
        "negative", "neutral", "positive"
    }
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6


def test_empty_text(monkeypatch):
    """Verify that /predict/text returns 422 Unprocessable Entity for blank or whitespace-only text."""
    setup_fake_model(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/predict/text", json={"text": "   "}
        )

    assert response.status_code == 422


def test_frontend_is_served():
    """Verify that the root endpoint '/' serves the single-page HTML frontend."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "FinDecision" in response.text


def test_model_info(monkeypatch):
    """Verify that /model-info returns metadata, labels, and local inference flag."""
    setup_fake_model(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["inference_local"] is True
