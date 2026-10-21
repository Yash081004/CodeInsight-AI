"""Adversarial and regression tests for specification-assisted assessment."""

import pytest

from backend.app.services.spec_hint_service import assess_spec, infer_spec_hint
from backend.app.services.inference_policy import (
    is_collapsed,
    route_classification,
    probability_margin,
)
from backend.app.config import ERROR_LABELS


SQUARE_SPEC = "Return the square of the given number."
SUM_INCLUSIVE = "Return the sum of all integers from 1 through n, inclusive."
ADULT_SPEC = "Return True if age is at least 18, otherwise return False."
SUM_EMPTY = "Return the sum of all elements in the list. The sum of an empty list should be zero."


@pytest.mark.parametrize(
    "code",
    [
        "def square(n):\n    return n * n",
        "def square(x):\n    return x * x",
        "def square(n):\n    return n**2",
        "def square(n):\n    return pow(n, 2)",
    ],
)
def test_square_correct_implementations(code):
    a = assess_spec(code, SQUARE_SPEC)
    assert a["status"] == "consistent"
    assert infer_spec_hint(code, SQUARE_SPEC) is None


def test_square_wrong_operator():
    code = "def square(n):\n    return n + n"
    a = assess_spec(code, SQUARE_SPEC)
    assert a["status"] == "violation"
    assert a["hint"]["label"] == "wrong_operator"
    assert infer_spec_hint(code, SQUARE_SPEC)["rule_id"] == "spec_square_vs_add"


def test_square_formatting_and_rename_still_wrong():
    code = "def square(num):\n    return num + num"
    assert assess_spec(code, SQUARE_SPEC)["status"] == "violation"


def test_suspicious_plus_but_not_square_spec():
    code = "def add(a, b):\n    return a + b"
    a = assess_spec(code, "Return the sum of two numbers.")
    assert a["status"] != "violation"


def test_vague_spec_no_violation():
    code = "def square(n):\n    return n + n"
    a = assess_spec(code, "fix square")
    assert a["status"] == "vague_spec"
    assert infer_spec_hint(code, "fix square") is None


def test_missing_problem_statement():
    code = "def square(n):\n    return n + n"
    assert assess_spec(code, None)["status"] == "no_statement"
    assert assess_spec(code, "   ")["status"] == "no_statement"


def test_inclusive_sum_correct():
    code = "def sum_to_n(n):\n    t = 0\n    for i in range(1, n + 1):\n        t += i\n    return t"
    assert assess_spec(code, SUM_INCLUSIVE)["status"] == "consistent"


def test_inclusive_sum_off_by_one():
    code = "def sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total"
    a = assess_spec(code, SUM_INCLUSIVE)
    assert a["status"] == "violation"
    assert a["hint"]["label"] == "off_by_one_loop"


def test_adult_correct():
    code = "def is_adult(age):\n    if age >= 18:\n        return True\n    return False"
    assert assess_spec(code, ADULT_SPEC)["status"] == "consistent"


def test_adult_inverted():
    code = "def is_adult(age):\n    if age < 18:\n        return True\n    return False"
    assert assess_spec(code, ADULT_SPEC)["status"] == "violation"


def test_sum_init_wrong():
    code = "def sum_values(values):\n    total = 1\n    for value in values:\n        total += value\n    return total"
    assert assess_spec(code, SUM_EMPTY)["status"] == "violation"


def test_sum_init_zero_not_violation():
    code = "def sum_values(values):\n    total = 0\n    for value in values:\n        total += value\n    return total"
    assert assess_spec(code, SUM_EMPTY)["status"] == "consistent"


def test_unsupported_spec_no_rule():
    code = "def foo():\n    return 1"
    a = assess_spec(code, "Implement binary search on a sorted array of integers.")
    assert a["status"] == "no_rule_applicable"
    assert infer_spec_hint(code, "Implement binary search on a sorted array of integers.") is None


def test_syntax_error_no_violation():
    a = assess_spec("def broken(\n  return 1", SQUARE_SPEC)
    assert a["status"] == "syntax_error"


def test_route_spec_only_when_collapsed_and_violation():
    flat_probs = {k: 0.2 for k in ERROR_LABELS}
    peaked = {k: 0.01 for k in ERROR_LABELS}
    peaked["wrong_operator"] = 0.96

    codebert_flat = {"predicted_label": "correct", "probabilities": flat_probs, "confidence": 0.21}
    codebert_peak = {"predicted_label": "wrong_operator", "probabilities": peaked, "confidence": 0.96}
    baseline = {"predicted_label": "wrong_operator", "probabilities": peaked, "confidence": 0.96}
    hint = {
        "label": "wrong_operator",
        "rule_id": "spec_square_vs_add",
        "confidence": "high",
        "message": "test",
    }
    assessment = {"status": "violation", "hint": hint}

    r, mt, _ = route_classification(
        codebert=codebert_flat,
        baseline=baseline,
        spec_hint=hint,
        spec_assessment=assessment,
        has_problem_statement=True,
    )
    assert mt == "spec_rule_assist"
    assert r["predicted_label"] == "wrong_operator"
    assert r["routing"]["score_applies_to_displayed_label"] is False

    r2, mt2, _ = route_classification(
        codebert=codebert_peak,
        baseline=baseline,
        spec_hint=hint,
        spec_assessment=assessment,
        has_problem_statement=True,
    )
    assert mt2 == "codebert_finetuned"
    assert r2["predicted_label"] == "wrong_operator"


def test_route_baseline_when_collapsed_and_separated():
    flat_probs = {k: 0.2 for k in ERROR_LABELS}
    bl_probs = {k: 0.01 for k in ERROR_LABELS}
    bl_probs["off_by_one_loop"] = 0.92
    codebert = {"predicted_label": "correct", "probabilities": flat_probs, "confidence": 0.21}
    baseline = {"predicted_label": "off_by_one_loop", "probabilities": bl_probs, "confidence": 0.92}
    r, mt, _ = route_classification(
        codebert=codebert,
        baseline=baseline,
        spec_hint=None,
        spec_assessment={"status": "no_statement"},
        has_problem_statement=False,
    )
    assert mt == "baseline_fallback"
    assert r["routing"]["primary_source"] == "baseline_fallback"
    assert is_collapsed(flat_probs)
    assert probability_margin(bl_probs) >= 0.10


def test_consistent_spec_does_not_trigger_assist():
    flat_probs = {k: 0.2 for k in ERROR_LABELS}
    codebert = {"predicted_label": "correct", "probabilities": flat_probs, "confidence": 0.21}
    r, mt, _ = route_classification(
        codebert=codebert,
        baseline=None,
        spec_hint=None,
        spec_assessment={"status": "consistent", "message": "ok"},
        has_problem_statement=True,
    )
    assert mt == "codebert_finetuned"
    assert r["routing"]["review_status"] == "inconclusive"
