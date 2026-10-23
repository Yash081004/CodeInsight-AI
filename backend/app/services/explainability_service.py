"""
Grounded diagnostics for classification (not attention-based causal claims).
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

from backend.app.config import ERROR_LABELS, FRIENDLY_LABELS


def _static_findings(code: str) -> List[Dict[str, str]]:
    """Deterministic AST/heuristic findings; each item is traceable to code structure."""
    findings: List[Dict[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        findings.append(
            {
                "kind": "ast",
                "severity": "error",
                "message": f"Syntax error: {e.msg} (line {e.lineno or '?'})",
            }
        )
        return findings

    findings.append({"kind": "ast", "severity": "info", "message": "Python syntax parse succeeded."})

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range" and len(node.iter.args) >= 2:
                    findings.append(
                        {
                            "kind": "static_heuristic",
                            "severity": "hint",
                            "message": (
                                "Loop uses range with explicit bounds — verify off-by-one "
                                "inclusive/exclusive endpoints against the problem statement."
                            ),
                        }
                    )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("result", "total", "acc", "product"):
                    if isinstance(node.value, ast.Constant) and node.value.value == 0:
                        if re.search(r"\*=|\*=|product|factorial", code, re.I):
                            findings.append(
                                {
                                    "kind": "static_heuristic",
                                    "severity": "hint",
                                    "message": (
                                        f"Accumulator `{target.id}` initialized to 0 before "
                                        "multiplication — multiplication identity is usually 1."
                                    ),
                                }
                            )

    if re.search(r"if\s+[^:]+\s*<\s*[^:]+\s*:\s*\n\s*return\s+\w+", code):
        findings.append(
            {
                "kind": "static_heuristic",
                "severity": "hint",
                "message": "Conditional return uses `<` — confirm max/min logic matches intent.",
            }
        )

    return findings


def build_classification_diagnostics(
    code: str,
    predicted_label: str,
    probabilities: Dict[str, float],
    *,
    dataset_behavioral_note: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    tests_executed: bool = False,
    test_results: Optional[List[Dict]] = None,
    routing: Optional[Dict[str, Any]] = None,
    codebert_snapshot: Optional[Dict[str, Any]] = None,
    baseline_snapshot: Optional[Dict[str, Any]] = None,
    spec_hint: Optional[Dict[str, Any]] = None,
    spec_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble explainability payload for API responses."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_alts = [
        {"label": lbl, "friendly_label": FRIENDLY_LABELS.get(lbl, lbl), "probability": round(p, 4)}
        for lbl, p in sorted_probs[1:4]
        if lbl in ERROR_LABELS
    ]

    max_p = max(probabilities.values()) if probabilities else 0.0
    second_p = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = max_p - second_p

    primary_source = (routing or {}).get("primary_source", "model_softmax")
    score_for_label = (routing or {}).get("score_applies_to_displayed_label", True)
    review_status = (routing or {}).get("review_status", "model_primary")

    uncertainty = "high"
    if primary_source == "spec_rule_assist":
        uncertainty = "spec_rule"
    elif not score_for_label:
        uncertainty = "high"
    elif margin >= 0.35:
        uncertainty = "low"
    elif margin >= 0.15:
        uncertainty = "medium"

    static = _static_findings(code)
    for f in static:
        f["source"] = "static_heuristic" if f.get("kind") == "static_heuristic" else f.get("kind", "static")

    limitation_notes: List[str] = []
    if task_context and task_context.get("requires_external_context"):
        limitation_notes.append(
            "Diagnosis may depend on problem statement or tests not provided in this request."
        )
    if not tests_executed:
        limitation_notes.append("No test cases were executed in this analysis.")

    collapse_warning = None
    if codebert_snapshot and routing:
        cb_margin = routing.get("codebert_margin")
        if cb_margin is not None and cb_margin < 0.05:
            collapse_warning = (
                "Production CodeBERT outputs are low-discrimination (near-uniform softmax). "
                "Held-out test macro F1 is ~13.3% (TF-IDF baseline ~27.2%). "
                "Do not treat the top class as a verified logical diagnosis."
            )

    known_model_limitations = [
        "Production CodeBERT fine-tune shows collapsed softmax on many inputs (margin often < 0.05).",
        "Reported evaluation: CodeBERT test macro F1 ≈ 13.3%; TF-IDF baseline ≈ 27.2%.",
        "Spec-assisted labels are deterministic rules, not trained classifier outputs.",
    ]

    evidence_parts: List[str] = []
    if codebert_snapshot:
        evidence_parts.append(
            f"CodeBERT would predict `{codebert_snapshot.get('predicted_label')}` "
            f"(margin {routing.get('codebert_margin') if routing else '?'})"
        )
    if baseline_snapshot:
        evidence_parts.append(
            f"TF-IDF baseline would predict `{baseline_snapshot.get('predicted_label')}`."
        )
    if spec_assessment and spec_assessment.get("status") not in (None, "no_statement"):
        evidence_parts.append(f"Spec assessment: {spec_assessment.get('status')} — {spec_assessment.get('message', '')}")
    if spec_hint:
        evidence_parts.append(f"Spec violation rule: {spec_hint.get('rule_id')}.")
    evidence_summary = " ".join(evidence_parts) if evidence_parts else "Model softmax only; no problem statement or spec rule applied."

    return {
        "prediction": {
            "label": predicted_label,
            "friendly_label": FRIENDLY_LABELS.get(predicted_label, predicted_label),
            "source": primary_source,
        },
        "routing": routing,
        "codebert_snapshot": {
            "predicted_label": codebert_snapshot.get("predicted_label"),
            "probabilities": codebert_snapshot.get("probabilities"),
            "logits": codebert_snapshot.get("logits"),
        }
        if codebert_snapshot
        else None,
        "baseline_snapshot": {
            "predicted_label": baseline_snapshot.get("predicted_label"),
            "probabilities": baseline_snapshot.get("probabilities"),
        }
        if baseline_snapshot
        else None,
        "spec_hint": spec_hint,
        "spec_assessment": spec_assessment,
        "collapse_warning": collapse_warning,
        "known_model_limitations": known_model_limitations,
        "review_status": review_status,
        "evidence_summary": evidence_summary,
        "score_applies_to_displayed_label": score_for_label,
        "confidence_score": round(max_p, 4),
        "uncertainty_band": uncertainty,
        "confidence_note": (
            "Softmax probability for the predicted class — not calibrated accuracy. "
            "When inference source is spec_rule_assist, scores reflect CodeBERT context only, "
            "not confidence in the spec-derived label."
            if primary_source == "spec_rule_assist"
            else "Softmax probability for the predicted class — not calibrated accuracy. "
            "Low margin between top classes indicates ambiguity."
        ),
        "top_alternative_labels": top_alts,
        "static_analysis": static,
        "static_analysis_disclaimer": (
            "Static hints are deterministic heuristics, not model explanations or proof of fault type."
        ),
        "behavioral_evidence": dataset_behavioral_note,
        "tests_executed": tests_executed,
        "test_results": test_results,
        "task_context_limitations": limitation_notes,
        "suggested_action": (
            "Review static hints separately from the model label. "
            "Provide a problem statement and run tests before treating the diagnosis as verified."
        ),
    }
