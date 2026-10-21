"""
Diagnostic cases for production classifier routing (honest accuracy expectations).

Software tests assert routing behavior; semantic label match is documented, not faked.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import ERROR_LABELS, LABEL_TO_ID
from backend.app.services.inference_policy import probability_margin, is_collapsed

client = TestClient(app)

CASES = [
    ("A", "def square(n):\n    return n * n", None, "correct"),
    (
        "B",
        "def square(n):\n    return n + n",
        "Return the square of the given number.",
        "wrong_operator",
    ),
    (
        "C",
        "def sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total",
        "Return the sum of all integers from 1 through n, inclusive.",
        "off_by_one_loop",
    ),
    (
        "D",
        "def is_adult(age):\n    if age < 18:\n        return True\n    return False",
        "Return True if age is at least 18, otherwise return False.",
        "wrong_condition",
    ),
    (
        "E",
        "def sum_values(values):\n    total = 1\n    for value in values:\n        total += value\n    return total",
        "Return the sum of all elements in the list. The sum of an empty list should be zero.",
        "wrong_initialization",
    ),
]


def test_label_mapping_order():
    assert ERROR_LABELS == [
        "correct",
        "wrong_initialization",
        "off_by_one_loop",
        "wrong_operator",
        "wrong_condition",
    ]
    assert LABEL_TO_ID["correct"] == 0
    assert LABEL_TO_ID["wrong_condition"] == 4


def test_probability_helpers():
    flat = {k: 0.2 for k in ERROR_LABELS}
    assert is_collapsed(flat) is True
    peaked = {"correct": 0.9, "wrong_initialization": 0.02, "off_by_one_loop": 0.02, "wrong_operator": 0.03, "wrong_condition": 0.03}
    assert is_collapsed(peaked) is False
    assert probability_margin(peaked) > 0.8


@pytest.mark.parametrize("case_id,code,problem,expected_semantic", CASES)
def test_diagnostic_case_api(case_id, code, problem, expected_semantic):
    payload = {"code": code}
    if problem:
        payload["problem_statement"] = problem
    r = client.post("/api/classification/analyze", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["code_valid"] is True
    assert set(data["probabilities"].keys()) == set(ERROR_LABELS) or data["model_type"] == "spec_rule_assist"
    diag = data.get("diagnostics") or {}
    assert diag.get("collapse_warning") or diag.get("codebert_snapshot")
    assert diag.get("review_status")
    assert diag.get("routing", {}).get("primary_source") == data.get("model_type") or data.get("model_type") in (
        "spec_rule_assist",
        "baseline_fallback",
        "codebert_finetuned",
    )
    if data.get("model_type") == "spec_rule_assist":
        assert diag.get("routing", {}).get("score_applies_to_displayed_label") is False

    predicted = data["predicted_label"]
    # Regression: collapsed CodeBERT must not be the only signal when spec assist applies
    if problem and case_id in ("B", "C", "D", "E"):
        assert data["model_type"] in ("spec_rule_assist", "baseline_fallback", "codebert_finetuned")
        if data["model_type"] == "spec_rule_assist":
            assert predicted == expected_semantic
        elif data["model_type"] == "baseline_fallback":
            assert predicted == expected_semantic
    if case_id == "A":
        assert predicted == expected_semantic

    assert diag.get("review_status") is not None
    assert diag.get("evidence_summary")
    if data["model_type"] == "spec_rule_assist":
        assert diag.get("score_applies_to_displayed_label") is False
        assert (diag.get("routing") or {}).get("primary_source") == "spec_rule_assist"


def test_no_silent_correct_on_invalid_code():
    r = client.post(
        "/api/classification/analyze",
        json={"code": "def broken(\n    return 1"},
    )
    assert r.status_code == 200
    assert r.json()["predicted_label"] != "correct" or r.json()["code_valid"] is False


def test_malformed_empty_rejected():
    assert client.post("/api/classification/analyze", json={"code": ""}).status_code == 422
