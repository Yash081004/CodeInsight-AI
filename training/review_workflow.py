"""
Shared human review workflow helpers (benchmark_v3).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.app.config import ERROR_LABELS

VALID_STATUS = frozenset({"pending", "accepted", "rejected", "relabeled", "needs_expert_review"})
PRIORITY_SAMPLE_IDS = frozenset({"S0027", "S0028", "S0003", "S0009"})


def decisions_path(dataset: str = "benchmark_v3") -> Path:
    if dataset == "benchmark_v3":
        return PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json"
    return PROJECT_ROOT / "data" / "benchmark_v2" / "review_decisions.json"


def load_decisions(dataset: str = "benchmark_v3") -> dict:
    p = decisions_path(dataset)
    if not p.exists():
        return {"dataset_version": dataset, "decisions": [], "human_reviewed": False}
    return json.loads(p.read_text(encoding="utf-8"))


def load_sample_row(sample_id: str) -> Optional[dict]:
    import pandas as pd

    csv = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["sample_id"] == sample_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def template_spec(source_template_id: str) -> dict:
    """Problem spec and template tests from generator registry (read-only)."""
    specs = {
        "factorial_1": {
            "problem_statement": "Compute factorial(n) for non-negative n; 0! = 1.",
            "template_tests": [
                "assert factorial(0) == 1",
                "assert factorial(1) == 1",
                "assert factorial(5) == 120",
                "assert factorial(3) == 6",
            ],
            "spec_source": "training/generate_dataset.py template factorial_1",
        },
        "factorial_3": {
            "problem_statement": "Compute factorial(n) using iterative multiplication from 2..n.",
            "template_tests": [
                "assert factorial(0) == 1",
                "assert factorial(1) == 1",
                "assert factorial(5) == 120",
                "assert factorial(4) == 24",
            ],
            "spec_source": "training/generate_dataset.py template factorial_3",
        },
        "palindrome_1": {
            "problem_statement": "Return True iff string s reads the same forward and backward.",
            "template_tests": [
                'assert is_palindrome("racecar") == True',
                'assert is_palindrome("hello") == False',
                'assert is_palindrome("aba") == True',
                'assert is_palindrome("a") == True',
                'assert is_palindrome("") == True',
            ],
            "spec_source": "training/generate_dataset.py template palindrome_1",
        },
    }
    return specs.get(
        source_template_id,
        {
            "problem_statement": None,
            "template_tests": [],
            "spec_source": None,
            "note": "No embedded template spec in registry — mark needs_expert_review if adjudicating.",
        },
    )


def reviewer_checklist() -> List[str]:
    return [
        "Does the code match the stated problem specification?",
        "Does the assigned fault type describe the actual bug (not merely a symptom)?",
        "Do template tests fail on this snippet? (Passing tests do not prove the label wrong.)",
        "Are review-only tests sufficient, or is expert judgment required?",
        "Is this sample suitable for a human-adjudicated benchmark export?",
    ]


def validate_decision_payload(
    status: str,
    rationale: Optional[str],
    new_label: Optional[str],
    reviewer_id: Optional[str],
    anonymous_mode: bool,
) -> Tuple[bool, str]:
    if status not in VALID_STATUS:
        return False, f"invalid status: {status}"
    if status == "pending":
        return False, "use decide only for non-pending transitions"
    if not reviewer_id and not anonymous_mode:
        return False, "provide --reviewer NAME or --anonymous-review"
    if status in ("rejected", "relabeled", "needs_expert_review") and not (rationale or "").strip():
        return False, f"rationale required for status={status}"
    if status == "relabeled":
        if not new_label or new_label not in ERROR_LABELS:
            return False, f"relabeled requires --new-label in {ERROR_LABELS}"
    if status == "accepted" and new_label:
        return False, "accepted confirms original label; do not pass --new-label"
    return True, "ok"


def build_review_packet(decision_row: dict, *, include_model_hints: bool = False) -> dict:
    sid = decision_row.get("sample_id")
    row = load_sample_row(sid) if sid else None
    spec = template_spec(decision_row.get("source_template_id") or (row or {}).get("source_template_id") or "")
    fixture_path = PROJECT_ROOT / "data" / "benchmark_v3" / "review_fixtures" / f"{sid}.json"
    fixture = {}
    if fixture_path.exists():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    packet = {
        "review_id": decision_row.get("review_id"),
        "sample_id": sid,
        "family_id": decision_row.get("family_id"),
        "source_template_id": decision_row.get("source_template_id"),
        "dataset_split": (row or {}).get("split"),
        "original_label": decision_row.get("original_label") or decision_row.get("current_label"),
        "current_label_in_csv": (row or {}).get("error_label"),
        "status": decision_row.get("status", "pending"),
        "problem_specification": spec,
        "source_code": (row or {}).get("code"),
        "mutation_type": decision_row.get("mutation_type") or (row or {}).get("mutation_type"),
        "mutation_rationale": decision_row.get("mutation_rationale") or (row or {}).get("mutation_description"),
        "dataset_validation_status": (row or {}).get("validation_status"),
        "expected_behavior_field": (row or {}).get("expected_behavior"),
        "behavioral_tests": {
            "template_tests": spec.get("template_tests", []),
            "review_only_tests": fixture.get("review_only_tests", []),
            "limitations": fixture.get("limitations", []),
            "execution_note": fixture.get(
                "execution_note",
                "Run: py -3.13 -m training.review_test_runner --sample-id <ID> (allowlisted samples only).",
            ),
        },
        "reviewer_checklist": reviewer_checklist(),
        "decision_fields": {
            "allowed_statuses": sorted(VALID_STATUS),
            "rationale_required_for": ["rejected", "relabeled", "needs_expert_review"],
            "reviewer_identity": "Set --reviewer or --anonymous-review on CLI; not auto-filled.",
        },
        "audit_note": "Recording a decision appends audit_history; dataset.csv is not modified.",
    }
    if include_model_hints:
        hints_path = PROJECT_ROOT / "artifacts" / "metrics" / "phase7" / "error_explorer_entries.json"
        if hints_path.exists() and sid:
            blob = json.loads(hints_path.read_text(encoding="utf-8"))
            for split in ("val", "test"):
                for model in ("codebert", "tfidf"):
                    for e in (blob.get(split) or {}).get(model) or []:
                        if e.get("sample_id") == sid:
                            packet.setdefault("optional_model_predictions_not_ground_truth", {})[
                                f"{model}_{split}"
                            ] = {
                                "predicted_label": e.get("predicted_label"),
                                "note": "Model output for analysis only — not ground truth.",
                            }
    return packet


def verify_workflow() -> dict:
    """Self-check for Phase 8 review infrastructure."""
    report: dict = {"ok": True, "checks": [], "errors": []}
    data = load_decisions("benchmark_v3")
    for d in data.get("decisions", []):
        if not d.get("review_id"):
            report["errors"].append("missing review_id")
        if d.get("sample_id") and not load_sample_row(d["sample_id"]):
            report["errors"].append(f"missing sample {d.get('sample_id')}")
        if d.get("original_label") != d.get("current_label") and d.get("status") == "pending":
            report["checks"].append(f"label drift recorded: {d.get('review_id')}")
    for sid in PRIORITY_SAMPLE_IDS:
        if not load_sample_row(sid):
            report["errors"].append(f"priority sample missing: {sid}")
    pkt = PROJECT_ROOT / "data" / "benchmark_v3" / "review_packets" / "priority_batch.json"
    report["priority_batch_present"] = pkt.exists()
    report["ok"] = len(report["errors"]) == 0
    report["verified_at"] = datetime.now(timezone.utc).isoformat()
    return report
