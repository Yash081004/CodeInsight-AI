"""
Build classifier training text from benchmark rows (no invented problem statements).

v3 `expected_behavior` is only passes_all_tests | fails_tests — not natural-language spec.
Spec+code mode uses factual `problem_type` as [TASK] context unless a real spec column is added later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SPEC_SEPARATOR = "\n[SEP]\n"
TASK_PREFIX = "[TASK] "
CODE_PREFIX = "[CODE] "

GENERIC_EXPECTED_BEHAVIOR = frozenset({"passes_all_tests", "fails_tests"})


def audit_specifications(df: pd.DataFrame) -> Dict[str, Any]:
    """Report specification field quality without modifying data."""
    out: Dict[str, Any] = {"row_count": len(df)}
    if "expected_behavior" in df.columns:
        eb = df["expected_behavior"].fillna("").astype(str).str.strip()
        out["expected_behavior_empty_count"] = int((eb == "").sum())
        out["expected_behavior_unique"] = sorted(eb.unique().tolist())
        out["expected_behavior_is_test_outcome_only"] = set(eb.unique()).issubset(
            GENERIC_EXPECTED_BEHAVIOR | {""}
        )
    if "problem_type" in df.columns:
        pt = df["problem_type"].fillna("").astype(str).str.strip()
        out["problem_type_empty_count"] = int((pt == "").sum())
        out["problem_type_unique_count"] = int(pt.nunique())
    if "mutation_description" in df.columns:
        md = df["mutation_description"].fillna("").astype(str).str.strip()
        out["mutation_description_empty_count"] = int((md == "").sum())
    out["has_natural_language_problem_statement_column"] = False
    out["notes"] = (
        "benchmark v3 stores test outcomes in expected_behavior, not user-facing problem statements. "
        "spec_code mode uses problem_type as non-invented task context."
    )
    return out


def build_model_input(row: Any, mode: str = "code_only") -> str:
    code = str(row["code"] if isinstance(row, dict) else row.code).strip()
    if mode == "code_only":
        return code
    if mode == "spec_code":
        task = str(
            row.get("problem_type") if isinstance(row, dict) else getattr(row, "problem_type", "")
        ).strip()
        if not task:
            task = "unknown_task"
        spec_part = f"{TASK_PREFIX}{task}"
        return f"{spec_part}{SPEC_SEPARATOR}{CODE_PREFIX}{code}"
    raise ValueError(f"Unknown input mode: {mode}")


def build_texts_from_frame(df: pd.DataFrame, mode: str) -> List[str]:
    return [build_model_input(row, mode) for row in df.to_dict("records")]


def truncate_strategy_doc(max_length: int) -> str:
    return (
        f"Single sequence tokenized with max_length={max_length}, truncation=True (tail truncated). "
        f"Format: [TASK] problem_type {SPEC_SEPARATOR.strip()} [CODE] source. "
        "Spec prefix is preserved when possible; long code may be truncated from the end."
    )
