# Simple test to check model pipeline import and FastAPI health
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from src.api.main import app

load_dotenv()

API_KEY = os.getenv("API_KEY")


def test_fastapi_health():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"title": "Concerns at school diploma plan"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code in (200, 503)  # 503 if model not trained
    if response.status_code == 200:
        data = response.json()
        assert "category" in data
        assert isinstance(data["category"], str)
        assert "confidence" in data
        assert (data["confidence"] is None) or (isinstance(data["confidence"], float))


def test_predict_missing_field():
    client = TestClient(app)
    response = client.post("/predict", json={}, headers={"X-API-Key": API_KEY})
    assert (
        response.status_code == 422
    )  # Unprocessable Entity for missing required field


def test_info_endpoint():
    client = TestClient(app)
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
    if data["model_loaded"]:
        assert "model_version" in data
        assert isinstance(data["model_version"], str)
        assert "classes" in data
        assert isinstance(data["classes"], list)


def test_predict_no_api_key():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"title": "Test title without API key"},
    )
    if response.status_code == 503:
        # Model not available, skip strict auth check
        assert response.json()["detail"] == "Model not available"
    else:
        assert response.status_code == 403
        assert response.json() == {"detail": "X-API-Key header missing"}


def test_predict_invalid_api_key():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"title": "Test title with invalid API key"},
        headers={"X-API-Key": "invalid_key"},
    )
    if response.status_code == 503:
        # Model not available, skip strict auth check
        assert response.json()["detail"] == "Model not available"
    else:
        assert response.status_code == 403
        assert response.json() == {"detail": "Invalid API key supplied"}
