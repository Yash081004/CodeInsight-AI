"""
Deterministic specification checks (problem statement + code structure).

These are NOT model predictions. Violations are only returned when a rule
positively matches both the stated spec and a contradictory code pattern.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Optional

from backend.app.config import ERROR_LABELS

_MIN_SPEC_WORDS = 4


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _is_vague_spec(spec: str) -> bool:
    if len(spec.split()) < _MIN_SPEC_WORDS:
        return True
    vague_markers = ("something", "fix it", "make it work", "as needed", "etc")
    if any(m in spec for m in vague_markers):
        return True
    return False


def _has_square_multiply_return(code: str) -> bool:
    compact = re.sub(r"\s+", "", code)
    if re.search(r"return\w+\*\w+", compact):
        return True
    if re.search(r"return\w+\*\*2", compact):
        return True
    if re.search(r"returnpow\(", compact, re.I):
        return True
    return False


def assess_spec(code: str, problem_statement: Optional[str]) -> Dict[str, Any]:
    """
    Returns assessment dict:
      status: no_statement | vague_spec | syntax_error | consistent | violation | no_rule_applicable
      hint: present only when status == violation (includes label, rule_id, confidence, message)
    """
    if not problem_statement or not problem_statement.strip():
        return {"status": "no_statement", "message": "No problem statement provided."}

    spec = _norm(problem_statement)
    if _is_vague_spec(spec):
        return {
            "status": "vague_spec",
            "message": "Problem statement is too vague for deterministic spec rules.",
        }

    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"status": "syntax_error", "message": f"Cannot assess spec against invalid syntax: {e.msg}"}

    # --- Square: positive (consistent) before violation ---
    if "square" in spec and re.search(r"def\s+square\s*\(", code):
        if _has_square_multiply_return(code):
            return {
                "status": "consistent",
                "message": "Implementation uses multiplication consistent with squaring.",
            }
        if re.search(r"return\s+\w+\s*\+\s*\w+\b", code):
            return {
                "status": "violation",
                "hint": {
                    "label": "wrong_operator",
                    "rule_id": "spec_square_vs_add",
                    "confidence": "high",
                    "message": "Problem asks for a square; implementation adds instead of multiplying.",
                },
                "message": "Spec rule: square implemented with addition.",
            }
        return {"status": "no_rule_applicable", "message": "Square spec present but no matched rule."}

    # --- Inclusive sum 1..n ---
    if ("sum" in spec or "1 through" in spec or "1 to n" in spec) and "inclusive" in spec:
        if re.search(r"range\s*\(\s*1\s*,\s*n\s*\+\s*1\s*\)", code.replace(" ", "")) or re.search(
            r"range\s*\(\s*1\s*,\s*n\s*\+\s*1\s*\)", code
        ):
            return {
                "status": "consistent",
                "message": "Loop range includes n (1 through n inclusive).",
            }
        if re.search(r"range\s*\(\s*1\s*,\s*n\s*\)", code):
            return {
                "status": "violation",
                "hint": {
                    "label": "off_by_one_loop",
                    "rule_id": "spec_inclusive_sum_range",
                    "confidence": "high",
                    "message": "Problem requires including n; loop uses range(1, n) which excludes n.",
                },
                "message": "Spec rule: off-by-one on inclusive upper bound.",
            }
        return {"status": "no_rule_applicable", "message": "Inclusive sum spec but loop pattern unmatched."}

    # --- Adult threshold ---
    if ("at least 18" in spec or "18 or older" in spec) and re.search(r"def\s+is_adult\s*\(", code):
        if re.search(r"if\s+age\s*>=\s*18\s*:\s*\n\s*return\s+True", code) or re.search(
            r"if\s+age\s*<\s*18\s*:\s*\n\s*return\s+False", code
        ):
            return {"status": "consistent", "message": "Adult threshold branch matches at-least-18 spec."}
        if re.search(r"if\s+age\s*<\s*18\s*:\s*\n\s*return\s+True", code):
            return {
                "status": "violation",
                "hint": {
                    "label": "wrong_condition",
                    "rule_id": "spec_adult_threshold_inverted",
                    "confidence": "high",
                    "message": "Problem requires True for age ≥ 18; branch returns True when age < 18.",
                },
                "message": "Spec rule: inverted adult threshold.",
            }
        return {"status": "no_rule_applicable", "message": "Adult spec present; condition pattern unmatched."}

    # --- Sum with empty list must be zero ---
    if "sum" in spec and "empty" in spec and "zero" in spec:
        if re.search(r"total\s*=\s*0\b", code) or re.search(r"=\s*0\b", code):
            return {"status": "consistent", "message": "Accumulator starts at zero per empty-sum spec."}
        if re.search(r"total\s*=\s*1\b", code) and re.search(r"\+=", code):
            return {
                "status": "violation",
                "hint": {
                    "label": "wrong_initialization",
                    "rule_id": "spec_sum_init_one",
                    "confidence": "high",
                    "message": "Problem defines empty-list sum as 0; accumulator starts at 1.",
                },
                "message": "Spec rule: wrong initialization for sum.",
            }
        return {"status": "no_rule_applicable", "message": "Sum/empty spec present; init pattern unmatched."}

    return {"status": "no_rule_applicable", "message": "No deterministic spec rule applies to this statement."}


def infer_spec_hint(code: str, problem_statement: Optional[str]) -> Optional[Dict[str, Any]]:
    """Backward-compatible: violation hint only."""
    assessment = assess_spec(code, problem_statement)
    if assessment.get("status") == "violation":
        hint = assessment.get("hint")
        if hint and hint.get("label") in ERROR_LABELS:
            return hint
    return None
