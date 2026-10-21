"""
Task contract: which fault types need external context beyond bare source code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Fault types that often need intended behavior to judge from code alone.
CONTEXT_BY_FAULT: Dict[str, dict] = {
    "correct": {
        "inferable_from_code_alone": "partial",
        "typical_missing": ["problem_statement", "test_cases", "expected_io"],
        "note": "Correctness requires knowing intended behavior; syntactically valid code is not sufficient.",
    },
    "wrong_initialization": {
        "inferable_from_code_alone": "often",
        "typical_missing": ["problem_statement"],
        "note": "Identity value errors (0 vs 1) are sometimes visible without spec if convention is clear.",
    },
    "off_by_one_loop": {
        "inferable_from_code_alone": "partial",
        "typical_missing": ["problem_statement", "inclusive/exclusive boundary spec"],
        "note": "Range endpoints need intended iteration semantics.",
    },
    "wrong_operator": {
        "inferable_from_code_alone": "partial",
        "typical_missing": ["problem_statement", "function_purpose"],
        "note": "Operator inversion (e.g. < vs >) requires knowing max vs min intent.",
    },
    "wrong_condition": {
        "inferable_from_code_alone": "partial",
        "typical_missing": ["problem_statement", "test_cases"],
        "note": "Predicate bugs (parity, bounds) need intended truth conditions.",
    },
}


def assess_task_context(
    label: str,
    *,
    code: str,
    expected_behavior: Optional[str] = None,
    problem_type: Optional[str] = None,
    problem_statement: Optional[str] = None,
    test_cases: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    meta = CONTEXT_BY_FAULT.get(label, CONTEXT_BY_FAULT["correct"])
    limitations: List[str] = []
    if not (problem_statement or "").strip():
        limitations.append("no_problem_statement_provided")
    if not (expected_behavior or "").strip():
        limitations.append("no_expected_behavior_in_request")
    if not test_cases:
        limitations.append("no_test_cases_executed")

    requires_external = label in ("correct", "wrong_condition", "off_by_one_loop", "wrong_operator") or (
        not expected_behavior and not problem_statement
    )

    return {
        "fault_type": label,
        "inferable_from_code_alone": meta["inferable_from_code_alone"],
        "requires_external_context": requires_external,
        "limitations": limitations,
        "guidance": meta["note"],
        "problem_type": problem_type,
        "has_expected_behavior_field": bool((expected_behavior or "").strip()),
        "has_problem_statement": bool((problem_statement or "").strip()),
        "tests_provided": bool(test_cases),
    }
