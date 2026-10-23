"""Regression tests for hybrid similarity decision policy."""

import pytest

from backend.app.services.similarity_policy import hybrid_midband_decision


def test_midband_rejects_sort_vs_search_pattern():
    """Documented false positive: high CodeBERT, mid TF-IDF."""
    d = hybrid_midband_decision(
        codebert_score=0.9905,
        tfidf_score=0.4995,
        threshold=0.8,
        tf_low=0.40,
        tf_confirm=0.55,
    )
    assert d.is_similar is False
    assert d.decision_model == "hybrid_rejected_mid_tfidf"


def test_rename_tolerance_low_tfidf():
    d = hybrid_midband_decision(
        codebert_score=0.9854,
        tfidf_score=0.3166,
        threshold=0.8,
        tf_low=0.40,
        tf_confirm=0.55,
    )
    assert d.is_similar is True
    assert d.decision_model == "hybrid_codebert_rename_tolerance"
    assert d.review_recommended is True


def test_confirmed_when_both_high():
    d = hybrid_midband_decision(
        codebert_score=1.0,
        tfidf_score=1.0,
        threshold=0.8,
    )
    assert d.is_similar is True
    assert d.decision_model == "hybrid_confirmed"


def test_codebert_below_threshold():
    d = hybrid_midband_decision(codebert_score=0.5, tfidf_score=0.9, threshold=0.8)
    assert d.is_similar is False


@pytest.mark.integration
def test_analyze_sort_vs_search_not_similar():
    from backend.app.services.similarity_service import analyze_similarity

    a = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""
    b = """def binary_search(arr, target):
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
    r = analyze_similarity(a, b, threshold=0.8)
    assert r["codebert_score"] is not None and r["codebert_score"] > 0.9
    assert r["is_similar"] is False
    assert r["decision_model"] == "hybrid_rejected_mid_tfidf"
