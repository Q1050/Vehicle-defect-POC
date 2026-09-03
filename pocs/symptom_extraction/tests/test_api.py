from fastapi.testclient import TestClient

from app import app


def test_health_and_extraction_endpoint():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/api/v1/diagnostics/symptoms", json={"text": "The brakes squeal when I stop."})
        assert response.status_code == 200
        body = response.json()
        assert {item["event"] for item in body["symptoms"]} >= {"brake_noise"}
        assert {item["condition"] for item in body["conditions"]} >= {"braking"}
