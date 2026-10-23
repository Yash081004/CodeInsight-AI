"""Error gallery API and serialization tests."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.classification_error_gallery import (
    _error_category,
    _serialize_code,
    build_gallery_entries,
    build_per_class_analysis,
)


class TestSerialization:
    def test_code_truncation_safe(self):
        long_code = "x = 1\n" * 2000
        out = _serialize_code(long_code, max_len=100)
        assert len(out) <= 100
        assert "truncated" in out

    def test_error_category_missed_fault(self):
        assert _error_category("wrong_operator", "correct") == "missed_fault"

    def test_gallery_matched_ids(self):
        base = [
            {
                "sample_id": "S1",
                "family_id": "f",
                "true_label": "correct",
                "predicted_label": "wrong_operator",
                "correct": False,
                "probabilities": {"correct": 0.1},
                "model_id": "b",
                "experiment_id": "e",
                "code": "pass",
                "error_category": "false_positive_fault",
            }
        ]
        cb = [
            {
                "sample_id": "S1",
                "family_id": "f",
                "true_label": "correct",
                "predicted_label": "correct",
                "correct": True,
                "probabilities": {"correct": 0.9},
                "model_id": "c",
                "experiment_id": "e2",
                "code": "pass",
                "error_category": "correct_prediction",
            }
        ]
        g = build_gallery_entries(base, cb)
        assert g["total"] == 1
        assert g["entries"][0]["baseline_only_wrong"] is False
        assert g["entries"][0]["codebert_only_wrong"] is True

    def test_per_class_analysis(self):
        recs = [
            {
                "true_label": "correct",
                "predicted_label": "wrong_operator",
                "correct": False,
            },
            {"true_label": "correct", "predicted_label": "correct", "correct": True},
        ]
        out = build_per_class_analysis(recs)
        assert out["correct"]["support"] == 2
        assert out["correct"]["errors"] == 1


class TestErrorGalleryAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.app.main import app

        return TestClient(app)

    def test_missing_run_returns_unavailable(self, client):
        r = client.get("/api/evaluation/error-gallery?run_id=benchmark_v2/nonexistent_run_xyz")
        assert r.status_code == 200
        assert r.json()["available"] is False

    def test_invalid_run_id_rejected(self, client):
        r = client.get("/api/evaluation/error-gallery?run_id=../secrets")
        assert r.status_code == 200
        assert r.json()["available"] is False

    def test_gallery_with_artifacts_if_present(self, client):
        bench = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2"
        if not bench.exists():
            pytest.skip("no benchmark runs")
        codebert_dirs = [p for p in bench.iterdir() if (p / "classification_error_gallery.json").exists()]
        if not codebert_dirs:
            pytest.skip("gallery not built yet")
        run_id = f"benchmark_v2/{codebert_dirs[0].name}"
        r = client.get(f"/api/evaluation/error-gallery?run_id={run_id}&correct=false&limit=5")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert "probability_note" in body
        for entry in body.get("entries", []):
            assert "code" in entry
            assert "<script" not in entry["code"].lower()
