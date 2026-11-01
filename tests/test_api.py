import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.pop("GEMINI_API_KEY", None)

from app import app


def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_reports_service_state():
    response = client().get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "trained_model_available" in payload


def test_demo_prediction_validates_ranges():
    response = client().post(
        "/api/predict",
        json={"gpa": 5, "attendance": 80, "studyHours": 10},
    )
    assert response.status_code == 400


def test_demo_prediction_identifies_mode():
    response = client().post(
        "/api/predict",
        json={"gpa": 3.5, "attendance": 90, "studyHours": 12},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "demo"
    assert 0 <= payload["confidence"] <= 100


def test_trained_endpoint_is_explicit_when_artifacts_missing():
    response = client().post("/api/predict-trained", json={})
    if response.status_code == 503:
        assert "not available" in response.get_json()["error"].lower()
    else:
        assert response.status_code == 400


def test_genai_endpoint_is_disabled_without_key():
    response = client().post("/api/genai/summary", json={"context": "test"})
    assert response.status_code == 503
