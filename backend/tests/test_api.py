"""
CodeInsight AI - Backend Tests

Tests for API endpoints, validation, and services.
Uses mocked model components - no GPU or model download required.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.config import ERROR_LABELS, FEEDBACK_TEMPLATES, FRIENDLY_LABELS

client = TestClient(app)


# ─── Health Endpoint ─────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "codebert_available" in data
        assert "classifier_available" in data
        assert "baseline_available" in data
        assert "dataset_available" in data
    
    def test_health_has_status_fields(self):
        response = client.get("/api/health")
        data = response.json()
        assert "classifier_status" in data
        assert "similarity_status" in data


# ─── Similarity Endpoint ────────────────────────────────────────────────────

class TestSimilarity:
    def test_similarity_valid_code(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "def add(a, b):\n    return a + b",
            "code_b": "def add(x, y):\n    return x + y",
        })
        assert response.status_code == 200
        data = response.json()
        assert "tfidf_score" in data or "codebert_score" in data
        assert "threshold" in data
        assert "explanation" in data
        assert "model_info" in data
    
    def test_similarity_empty_code_a(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "",
            "code_b": "def f(): pass",
        })
        assert response.status_code == 422  # validation error
    
    def test_similarity_whitespace_only(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "   ",
            "code_b": "def f(): pass",
        })
        assert response.status_code == 422
    
    def test_similarity_with_threshold(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "def f(n):\n    return n * 2",
            "code_b": "def g(n):\n    return n * 2",
            "threshold": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.5
    
    def test_similarity_invalid_threshold(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "def f(): pass",
            "code_b": "def g(): pass",
            "threshold": 1.5,
        })
        assert response.status_code == 422
    
    def test_similarity_long_code_rejected(self):
        response = client.post("/api/similarity/analyze", json={
            "code_a": "x" * 10001,
            "code_b": "def f(): pass",
        })
        assert response.status_code == 422


# ─── Classification Endpoint ────────────────────────────────────────────────

class TestClassification:
    def test_classification_valid_code(self):
        response = client.post("/api/classification/analyze", json={
            "code": "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
        })
        assert response.status_code == 200
        data = response.json()
        # Model may or may not be available, but response should be valid
        assert "predicted_label" in data
        assert "feedback" in data
        assert "model_type" in data
    
    def test_classification_empty_code(self):
        response = client.post("/api/classification/analyze", json={
            "code": "",
        })
        assert response.status_code == 422
    
    def test_classification_whitespace_only(self):
        response = client.post("/api/classification/analyze", json={
            "code": "  \n  ",
        })
        assert response.status_code == 422


# ─── Dataset Endpoint ───────────────────────────────────────────────────────

class TestDatasets:
    def test_dataset_summary(self):
        response = client.get("/api/datasets/summary")
        assert response.status_code == 200
        data = response.json()
        assert "classification_available" in data
        assert "similarity_available" in data
        assert "total_classification_samples" in data
    
    def test_dataset_samples(self):
        response = client.get("/api/datasets/samples?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert "total" in data
        assert "page" in data
    
    def test_dataset_samples_invalid_page(self):
        response = client.get("/api/datasets/samples?page=0")
        assert response.status_code == 422
    
    def test_generate_without_confirm(self):
        response = client.post("/api/datasets/generate", json={
            "confirm": False,
        })
        assert response.status_code == 400


# ─── Evaluation Endpoint ────────────────────────────────────────────────────

class TestEvaluation:
    def test_evaluation_summary(self):
        response = client.get("/api/evaluation/summary")
        assert response.status_code == 200
        data = response.json()
        assert "classification_available" in data
        assert "similarity_available" in data
    
    def test_confusion_matrix(self):
        response = client.get("/api/evaluation/confusion-matrix")
        assert response.status_code == 200


# ─── History Endpoint ───────────────────────────────────────────────────────

class TestHistory:
    def test_get_history(self):
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data
    
    def test_clear_history_without_confirm(self):
        response = client.post("/api/history/clear?confirm=false")
        assert response.status_code == 400


# ─── Config and Constants ───────────────────────────────────────────────────

class TestConfig:
    def test_error_labels_count(self):
        assert len(ERROR_LABELS) == 5
    
    def test_error_labels_values(self):
        expected = {"correct", "wrong_initialization", "off_by_one_loop",
                     "wrong_operator", "wrong_condition"}
        assert set(ERROR_LABELS) == expected
    
    def test_feedback_templates_complete(self):
        for label in ERROR_LABELS:
            assert label in FEEDBACK_TEMPLATES
            assert len(FEEDBACK_TEMPLATES[label]) > 0
    
    def test_friendly_labels_complete(self):
        for label in ERROR_LABELS:
            assert label in FRIENDLY_LABELS


# ─── Code Validator ─────────────────────────────────────────────────────────

class TestCodeValidator:
    def test_valid_python(self):
        from backend.app.services.code_validator import validate_python_code
        is_valid, error = validate_python_code("def f(): pass")
        assert is_valid
        assert error == ""
    
    def test_invalid_python(self):
        from backend.app.services.code_validator import validate_python_code
        is_valid, error = validate_python_code("def f(")
        assert not is_valid
        assert "Syntax error" in error
    
    def test_empty_code(self):
        from backend.app.services.code_validator import validate_python_code
        is_valid, error = validate_python_code("")
        assert not is_valid
    
    def test_max_length(self):
        from backend.app.services.code_validator import validate_python_code
        is_valid, error = validate_python_code("x" * 100, max_length=50)
        assert not is_valid
        assert "maximum length" in error


# ─── Feedback Service ───────────────────────────────────────────────────────

class TestFeedbackService:
    def test_all_labels_have_feedback(self):
        from backend.app.services.feedback_service import get_feedback
        for label in ERROR_LABELS:
            feedback = get_feedback(label)
            assert len(feedback) > 0
    
    def test_unknown_label(self):
        from backend.app.services.feedback_service import get_feedback
        feedback = get_feedback("nonexistent")
        assert "Unknown label" in feedback
    
    def test_friendly_labels(self):
        from backend.app.services.feedback_service import get_friendly_label
        assert get_friendly_label("correct") == "Correct Code"
        assert get_friendly_label("off_by_one_loop") == "Off-by-One Loop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
