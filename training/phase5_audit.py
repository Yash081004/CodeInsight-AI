"""
Phase 5 reproducible audit for benchmark_v2/v3 and metric artifacts.

Usage:
    py -3.13 -m training.phase5_audit
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL
from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_pipeline.splits_v3 import class_coverage_by_split, families_per_split
from training.dataset_validation import sha256_file


REQUIRED_COLS = {
    "sample_id",
    "code",
    "error_label",
    "split",
    "family_id",
    "source_template_id",
    "dataset_version",
}


def _audit_dataset(version: str) -> dict:
    root = PROJECT_ROOT / "data" / version
    csv = root / "classification" / "dataset.csv"
    if not csv.exists():
        return {"version": version, "present": False}
    checksum = sha256_file(csv)
    df = pd.read_csv(csv)
    missing = REQUIRED_COLS - set(df.columns)
    errors = []
    if missing:
        errors.append(f"missing_columns={sorted(missing)}")
    invalid = set(df["error_label"]) - set(ERROR_LABELS)
    if invalid:
        errors.append(f"invalid_labels={invalid}")
    for lid, label in enumerate(ERROR_LABELS):
        if LABEL_TO_ID[label] != lid or ID_TO_LABEL[lid] != label:
            errors.append("label_id_mapping_drift")
            break
    records = df.to_dict("records")
    ok, iso = verify_group_isolation(records, "family_id")
    if not ok:
        errors.extend(iso[:5])
    dup_ids = df["sample_id"].duplicated().sum()
    if dup_ids:
        errors.append(f"duplicate_sample_ids={int(dup_ids)}")
    review = root / "classification" / "review_queue.json"
    rq = json.loads(review.read_text()) if review.exists() else []
    return {
        "version": version,
        "present": True,
        "checksum_sha256": checksum,
        "schema_columns": list(df.columns),
        "sample_count": len(df),
        "label_counts_overall": {k: int(v) for k, v in df["error_label"].value_counts().items()},
        "label_counts_by_split": {
            sp: {k: int(v) for k, v in df[df["split"] == sp]["error_label"].value_counts().items()}
            for sp in ("train", "val", "test")
        },
        "families_per_split": families_per_split(records) if "family_id" in df.columns else {},
        "family_count_by_split": {k: len(v) for k, v in families_per_split(records).items()},
        "class_coverage_by_split": class_coverage_by_split(records),
        "review_queue_count": len(rq),
        "errors": errors,
        "integrity_ok": len(errors) == 0 and ok,
    }


def _artifact_lineage() -> dict:
    paths = {
        "v2_tfidf_baseline": PROJECT_ROOT / "artifacts/metrics/benchmark_v2/20261019_062841/classification_metrics.json",
        "v2_codebert_phase3": PROJECT_ROOT / "artifacts/metrics/benchmark_v2/codebert_20261019_082413/codebert_classification_metrics.json",
        "phase4_audit": PROJECT_ROOT / "artifacts/metrics/benchmark_v2/phase4/collapse_audit.json",
        "ablation_summary": PROJECT_ROOT / "artifacts/metrics/benchmark_v2/ablations/ablation_summary.json",
        "production_checkpoint": PROJECT_ROOT / "models/codebert_classifier/config.json",
    }
    out = {}
    for k, p in paths.items():
        if p.exists():
            data = json.loads(p.read_text()) if p.suffix == ".json" else {"exists": True}
            out[k] = {"path": str(p.relative_to(PROJECT_ROOT)), "summary": data if isinstance(data, dict) else {}}
        else:
            out[k] = {"path": str(p.relative_to(PROJECT_ROOT)), "missing": True}
    return out


def _predicted_distributions() -> dict:
    dist = {}
    pred_files = [
        ("v2_codebert_test", "artifacts/metrics/benchmark_v2/codebert_20261019_082413/codebert_test_predictions.json"),
        ("v2_tfidf_test", "artifacts/metrics/benchmark_v2/20261019_062841/test_predictions.json"),
    ]
    for name, rel in pred_files:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        dist[name] = dict(Counter(r.get("predicted_label") for r in rows))
    return dist


def main() -> int:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_mapping": {"label_to_id": LABEL_TO_ID},
        "datasets": {
            "benchmark_v2": _audit_dataset("benchmark_v2"),
            "benchmark_v3": _audit_dataset("benchmark_v3"),
        },
        "artifact_lineage": _artifact_lineage(),
        "predicted_label_distributions": _predicted_distributions(),
        "historical_test_disclaimer": "benchmark_v2 test split was previously inspected; not an untouched holdout.",
    }
    out_dir = PROJECT_ROOT / "artifacts/metrics/phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    failed = []
    for v, d in report["datasets"].items():
        if d.get("present") and not d.get("integrity_ok"):
            failed.append(v)
    print(f"[OK] wrote {out_path}")
    if failed:
        print(f"[FAIL] integrity: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
