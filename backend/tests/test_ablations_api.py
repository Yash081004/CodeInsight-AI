"""Ablation listing API tests."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_ablations_endpoint_shape():
    r = client.get("/api/evaluation/ablations")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data
    assert "experiments" in data
