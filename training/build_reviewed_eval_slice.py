"""
Export human-adjudicated benchmark_v3 samples into data/benchmark_reviewed/ (when available).

Does not modify dataset.csv or invent labels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file
from training.review_workflow import VALID_STATUS


def _blocked_manifest(reason: str, **extra) -> dict:
    return {
        "status": "blocked_pending_human_review",
        "human_reviewed_sample_count": 0,
        "eval_slice_path": None,
        "blocked_reason": reason,
        "next_action": "Record accepted or relabeled decisions via label_review_cli decide, then re-run this script.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def main() -> int:
    decisions_path = PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json"
    manifest_path = PROJECT_ROOT / "data" / "benchmark_reviewed" / "manifest.json"
    v3_csv = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not decisions_path.exists():
        print("[BLOCKED] No review_decisions.json")
        manifest_path.write_text(json.dumps(_blocked_manifest("missing_review_decisions"), indent=2), encoding="utf-8")
        return 2

    data = json.loads(decisions_path.read_text(encoding="utf-8"))
    eligible = []
    for d in data.get("decisions", []):
        st = d.get("status")
        if st not in VALID_STATUS:
            print(f"[FAIL] invalid status on {d.get('review_id')}")
            return 1
        if st in ("pending", "needs_expert_review", "rejected"):
            continue
        if st == "relabeled" and not d.get("proposed_label"):
            print(f"[FAIL] relabeled without proposed_label: {d.get('review_id')}")
            return 1
        if st in ("accepted", "relabeled") and not (d.get("rationale") or "").strip():
            print(f"[FAIL] missing rationale for export: {d.get('review_id')}")
            return 1
        if st != "pending" and not d.get("reviewer_id"):
            print(f"[FAIL] missing reviewer_id: {d.get('review_id')}")
            return 1
        if st in ("accepted", "relabeled"):
            eligible.append(d)

    if not eligible:
        print("[BLOCKED] Zero accepted/relabeled decisions — credible reviewed eval unavailable.")
        manifest_path.write_text(
            json.dumps(_blocked_manifest("zero_adjudicated_samples"), indent=2),
            encoding="utf-8",
        )
        return 0

    import pandas as pd

    df = pd.read_csv(v3_csv)
    train_families = set(df[df["split"] == "train"]["family_id"])
    rows = []
    for d in eligible:
        sid = d.get("sample_id")
        if not sid:
            print(f"[FAIL] missing sample_id on {d.get('review_id')}")
            return 1
        row = df[df["sample_id"] == sid]
        if row.empty:
            print(f"[FAIL] unknown sample_id {sid}")
            return 1
        fam = row.iloc[0]["family_id"]
        if fam in train_families:
            print(f"[FAIL] family overlap with train: {fam} ({sid})")
            return 1
        label = d["proposed_label"] if d.get("status") == "relabeled" else row.iloc[0]["error_label"]
        rows.append(
            {
                **row.iloc[0].to_dict(),
                "adjudicated_label": label,
                "review_id": d["review_id"],
                "review_status": d["status"],
                "reviewer_id": d.get("reviewer_id"),
                "review_rationale": d.get("rationale"),
                "decided_at": d.get("decided_at"),
                "original_csv_label": row.iloc[0]["error_label"],
                "specification_status": "human_adjudicated",
            }
        )

    if not rows:
        print("[BLOCKED] No exportable rows after overlap filter")
        manifest_path.write_text(json.dumps(_blocked_manifest("overlap_or_empty"), indent=2), encoding="utf-8")
        return 2

    out_csv = PROJECT_ROOT / "data" / "benchmark_reviewed" / "classification" / "dataset.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    checksum = sha256_file(out_csv)
    ok, issues = verify_group_isolation(rows, "family_id")
    manifest = {
        "status": "ready",
        "human_reviewed_sample_count": len(rows),
        "eval_slice_path": str(out_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checksum_sha256": checksum,
        "no_train_family_overlap": ok,
        "integrity": issues[:5],
        "provenance": {
            "source_dataset": "benchmark_v3",
            "source_checksum": sha256_file(v3_csv) if v3_csv.exists() else None,
            "export_policy": "accepted and relabeled only; pending/rejected/needs_expert_review excluded",
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] exported {len(rows)} reviewed samples; checksum={checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
