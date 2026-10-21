"""Similarity API response contract (hybrid policy fields)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

BUBBLE = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""

BINARY = """def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1"""


def test_similarity_response_schema_fields():
    r = client.post(
        "/api/similarity/analyze",
        json={"code_a": "def f(): pass", "code_b": "def g(): pass", "threshold": 0.8},
    )
    assert r.status_code == 200
    data = r.json()
    for key in (
        "codebert_available",
        "tfidf_available",
        "threshold",
        "decision_model",
        "explanation",
        "model_info",
        "code_a_valid",
        "code_b_valid",
    ):
        assert key in data
    assert isinstance(data.get("review_recommended"), bool)


@pytest.mark.integration
def test_sort_vs_search_not_similar_under_hybrid():
    r = client.post(
        "/api/similarity/analyze",
        json={"code_a": BUBBLE, "code_b": BINARY, "threshold": 0.8},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_similar"] is False
    assert "hybrid" in data["decision_model"] or data["decision_model"] == "codebert"
