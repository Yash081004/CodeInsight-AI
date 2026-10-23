"""Spec+code training input construction."""

import pandas as pd

from backend.app.config import ERROR_LABELS
from training.phase10_v3_exclusions import apply_train_val_template_exclusion, load_review_queue_template_ids
from training.spec_code_input import (
    SPEC_SEPARATOR,
    TASK_PREFIX,
    audit_specifications,
    build_model_input,
    build_texts_from_frame,
)


def test_build_spec_code_format():
    row = {"code": "def f():\n    pass", "problem_type": "factorial"}
    text = build_model_input(row, "spec_code")
    assert text.startswith(TASK_PREFIX + "factorial")
    assert SPEC_SEPARATOR.strip() in text
    assert "def f()" in text


def test_code_only_unchanged():
    row = {"code": "x = 1", "problem_type": "gcd"}
    assert build_model_input(row, "code_only") == "x = 1"


def test_audit_flags_no_natural_language_spec():
    df = pd.DataFrame(
        [
            {
                "code": "def a(): pass",
                "problem_type": "gcd",
                "expected_behavior": "passes_all_tests",
                "mutation_description": "Original",
            }
        ]
    )
    audit = audit_specifications(df)
    assert audit["expected_behavior_is_test_outcome_only"] is True
    assert audit["has_natural_language_problem_statement_column"] is False


def test_filtered_view_excludes_queue_from_train_val():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    v3 = root / "data/benchmark_v3/classification/dataset.csv"
    queue = root / "data/benchmark_v3/classification/review_queue.json"
    df = pd.read_csv(v3)
    tids = load_review_queue_template_ids(queue)
    filtered, excluded = apply_train_val_template_exclusion(df, tids)
    assert (excluded["split"] != "test").all()
    test_ids_before = set(df[df["split"] == "test"]["sample_id"])
    test_ids_after = set(filtered[filtered["split"] == "test"]["sample_id"])
    assert test_ids_before == test_ids_after


def test_label_ordering_in_frame_build():
    df = pd.DataFrame(
        {"code": ["a", "b"], "problem_type": ["x", "y"], "error_label": ["correct", "wrong_operator"]}
    )
    texts = build_texts_from_frame(df, "spec_code")
    assert len(texts) == 2
    assert list(ERROR_LABELS)[0] == "correct"
