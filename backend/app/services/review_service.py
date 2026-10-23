"""Human review queue and reviewed-slice status (read-only; no label mutation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.config import PROJECT_ROOT, get_settings

VALID_STATUS = frozenset({"pending", "accepted", "rejected", "relabeled", "needs_expert_review"})
ADJUDICATED_STATUSES = frozenset({"accepted", "rejected", "relabeled", "needs_expert_review"})
EXPORT_ELIGIBLE_STATUSES = frozenset({"accepted", "relabeled"})


def _project_root() -> Path:
    return PROJECT_ROOT


def load_v3_review_decisions() -> dict:
    path = _project_root() / "data" / "benchmark_v3" / "review_decisions.json"
    if not path.exists():
        return {
            "available": False,
            "dataset_version": "benchmark_v3",
            "human_reviewed": False,
            "adjudication_complete": False,
            "decisions": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data["available"] = True
    return data


def load_reviewed_slice_manifest() -> dict:
    path = _project_root() / "data" / "benchmark_reviewed" / "manifest.json"
    if not path.exists():
        return {
            "available": False,
            "status": "blocked_pending_human_review",
            "human_reviewed_sample_count": 0,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data["available"] = True
    return data


def review_status_for_sample(sample_id: str) -> Optional[dict]:
    data = load_v3_review_decisions()
    for d in data.get("decisions") or []:
        if d.get("sample_id") == sample_id:
            return {
                "review_id": d.get("review_id"),
                "status": d.get("status"),
                "human_reviewed": d.get("status") not in (None, "pending"),
            }
    return None


def audit_integrity() -> List[str]:
    """Return integrity warnings (empty if OK)."""
    issues: List[str] = []
    data = load_v3_review_decisions()
    for d in data.get("decisions") or []:
        st = d.get("status")
        if st not in VALID_STATUS:
            issues.append(f"invalid_status:{d.get('review_id')}")
        if st in ("rejected", "relabeled", "needs_expert_review") and not d.get("rationale"):
            issues.append(f"missing_rationale:{d.get('review_id')}")
        if st == "relabeled" and not d.get("proposed_label"):
            issues.append(f"missing_proposed_label:{d.get('review_id')}")
        if st != "pending" and not d.get("reviewer_id"):
            issues.append(f"non_pending_without_reviewer:{d.get('review_id')}")
    if data.get("human_reviewed") and all(d.get("status") == "pending" for d in data.get("decisions") or []):
        issues.append("human_reviewed_flag_true_but_all_pending")
    return issues


def _train_families() -> set:
    import pandas as pd

    csv = _project_root() / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    return set(df[df["split"] == "train"]["family_id"])


def export_eligibility(decision: dict, train_families: Optional[set] = None) -> Dict[str, Any]:
    """Whether a decision can be included in benchmark_reviewed export (does not export)."""
    train_families = train_families if train_families is not None else _train_families()
    st = decision.get("status")
    reasons: List[str] = []
    if st in ("pending", "needs_expert_review"):
        reasons.append(f"status_{st}")
    if st == "rejected":
        reasons.append("status_rejected")
    if st == "relabeled" and not decision.get("proposed_label"):
        reasons.append("missing_proposed_label")
    if st in EXPORT_ELIGIBLE_STATUSES and not (decision.get("rationale") or "").strip():
        reasons.append("missing_rationale")
    if st != "pending" and not decision.get("reviewer_id"):
        reasons.append("missing_reviewer_id")
    fam = decision.get("family_id")
    if fam and fam in train_families:
        reasons.append("train_family_overlap")
    eligible = st in EXPORT_ELIGIBLE_STATUSES and len(reasons) == 0
    return {"export_eligible": eligible, "reasons": reasons}


def build_review_lab_summary(limit: int = 100, priority_only: bool = False) -> Dict[str, Any]:
    data = load_v3_review_decisions()
    manifest = load_reviewed_slice_manifest()
    decisions = data.get("decisions") or []
    train_fam = _train_families()

    if priority_only:
        priority_ids = {"S0027", "S0028", "S0003", "S0009"}
        decisions = [d for d in decisions if d.get("sample_id") in priority_ids]

    pending = sum(1 for d in decisions if d.get("status") == "pending")
    adjudicated = sum(1 for d in decisions if d.get("status") in ADJUDICATED_STATUSES)
    export_eligible = 0
    items = []
    for d in decisions[:limit]:
        elig = export_eligibility(d, train_fam)
        if elig["export_eligible"]:
            export_eligible += 1
        items.append(
            {
                "review_id": d.get("review_id"),
                "sample_id": d.get("sample_id"),
                "family_id": d.get("family_id"),
                "current_label": d.get("current_label") or d.get("original_label"),
                "original_label": d.get("original_label"),
                "proposed_label": d.get("proposed_label"),
                "status": d.get("status"),
                "rationale": d.get("rationale"),
                "reviewer_id": d.get("reviewer_id"),
                "decided_at": d.get("decided_at"),
                "expected_behavior": d.get("expected_behavior"),
                "mutation_rationale": d.get("mutation_rationale"),
                "audit_history_count": len(d.get("audit_history") or []),
                "audit_history": d.get("audit_history") or [],
                "export_eligibility": elig,
            }
        )

    slice_blocked = manifest.get("status") != "ready" or manifest.get("human_reviewed_sample_count", 0) == 0
    return {
        "available": data.get("available", False),
        "dataset_version": "benchmark_v3",
        "human_reviewed": data.get("human_reviewed", False),
        "adjudication_complete": data.get("adjudication_complete", False),
        "counts": {
            "total_queue": len(data.get("decisions") or []),
            "total_in_view": len(items),
            "pending": pending,
            "adjudicated": adjudicated,
            "export_eligible": export_eligible,
        },
        "items": items,
        "integrity_warnings": audit_integrity(),
        "reviewed_eval_slice": {
            "status": manifest.get("status"),
            "human_reviewed_sample_count": manifest.get("human_reviewed_sample_count", 0),
            "eval_slice_path": manifest.get("eval_slice_path"),
            "checksum_sha256": manifest.get("checksum_sha256"),
            "blocked": slice_blocked,
            "blocked_reason": manifest.get("blocked_reason"),
            "next_action": manifest.get(
                "next_action",
                "Record accepted or relabeled decisions via label_review_cli decide, then run build_reviewed_eval_slice.",
            ),
        },
        "synthetic_until_adjudicated": True,
        "note": "Labels remain synthetic until a human records accepted/relabeled with reviewer identity.",
    }
