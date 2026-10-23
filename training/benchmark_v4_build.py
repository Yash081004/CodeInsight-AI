"""
Benchmark v4 build infrastructure (read-only on v3 labels).

Eligible rows require human-reviewed family specifications. Others are quarantined
with reasons; v3 files are never modified.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file
from training.phase10_v3_exclusions import load_review_queue_template_ids
from training.spec_catalog import load_catalog

V3_CSV = PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv"
V3_QUEUE = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
V4_ROOT = PROJECT_ROOT / "data/benchmark_v4"
DATASET_VERSION = "benchmark_v4"


def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def _hash(code: str) -> str:
    return hashlib.sha256(_norm(code).encode()).hexdigest()


def human_reviewed_families(catalog: Dict[str, Any]) -> Set[str]:
    fams: Set[str] = set()
    for e in catalog.get("entries") or []:
        if (e.get("provenance") or {}).get("review_status") == "human_reviewed":
            fams.add(str(e.get("family_id")))
    return fams


def quarantine_reasons_for_row(
    row: Dict[str, Any],
    queue_template_ids: Set[str],
    reviewed_fams: Set[str],
) -> List[str]:
    reasons: List[str] = []
    tid = str(row.get("source_template_id", ""))
    fam = str(row.get("family_id", ""))
    mut = str(row.get("mutation_type", ""))
    label = str(row.get("error_label", ""))

    if tid in queue_template_ids:
        reasons.append("review_queue_template")
    if fam not in reviewed_fams:
        reasons.append("family_spec_not_human_reviewed")
    if mut == "style_variant":
        reasons.append("synthetic_style_variant")
    if label == "correct" and row.get("expected_behavior") == "passes_all_tests":
        reasons.append("correct_baseline_requires_spec_context")
    return reasons


def build_v4(
    *,
    catalog_path: Path | None = None,
    write_csv: bool = True,
) -> Dict[str, Any]:
    catalog = load_catalog(catalog_path)
    reviewed = human_reviewed_families(catalog)
    queue_tids = load_review_queue_template_ids(V3_QUEUE)

    df = pd.read_csv(V3_CSV)
    source_checksum = sha256_file(V3_CSV)

    eligible_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rd = row.to_dict()
        reasons = quarantine_reasons_for_row(rd, queue_tids, reviewed)
        entry = {**rd, "quarantine_reasons": reasons, "v4_provenance": "benchmark_v3_inherited"}
        if reasons:
            quarantine_rows.append(entry)
        else:
            eligible_rows.append(entry)

    eligible_df = pd.DataFrame(eligible_rows) if eligible_rows else pd.DataFrame(columns=df.columns)
    quarantine_df = pd.DataFrame(quarantine_rows)

    iso_ok, iso_errs = verify_group_isolation(eligible_df.to_dict("records"), "family_id") if len(eligible_df) else (True, [])

    cross_split_dups: List[Dict[str, Any]] = []
    if len(eligible_df):
        hashes = eligible_df["code"].map(lambda c: _hash(str(c)))
        eligible_df = eligible_df.copy()
        eligible_df["_code_hash"] = hashes
        by_hash: Dict[str, List[str]] = defaultdict(list)
        by_split: Dict[str, str] = {}
        for _, r in eligible_df.iterrows():
            h = r["_code_hash"]
            by_hash[h].append(str(r["sample_id"]))
            by_split[str(r["sample_id"])] = str(r["split"])
        for h, sids in by_hash.items():
            splits = {by_split[s] for s in sids}
            if len(splits) > 1:
                cross_split_dups.append({"code_hash_sha256": h, "sample_ids": sids, "splits": sorted(splits)})

    class_by_split: Dict[str, Dict[str, int]] = {}
    if len(eligible_df):
        for sp in sorted(eligible_df["split"].unique()):
            class_by_split[sp] = Counter(eligible_df[eligible_df["split"] == sp]["error_label"].astype(str))

    test_df = eligible_df[eligible_df["split"] == "test"] if len(eligible_df) and "split" in eligible_df.columns else pd.DataFrame()
    test_frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "source_v3_checksum_sha256": source_checksum,
        "test_row_count": int(len(test_df)),
        "test_sample_ids": test_df["sample_id"].tolist() if len(test_df) else [],
        "test_family_ids": sorted(test_df["family_id"].dropna().unique().tolist()) if len(test_df) else [],
        "test_inspected": True,
        "pristine_claim": False,
        "pristine_claim_note": "v3-derived; human specs and queue adjudication incomplete at freeze time.",
    }
    test_frozen["checksum_sha256"] = hashlib.sha256(
        json.dumps(test_frozen, sort_keys=True).encode()
    ).hexdigest()

    manifest: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_benchmark_v3_csv_checksum_sha256": source_checksum,
        "human_reviewed": False,
        "eligibility_rules": [
            "family has human_reviewed spec in catalog",
            "source_template_id not on review_queue template list",
            "row not synthetic style_variant",
            "correct rows quarantined until spec context policy defined",
        ],
        "eligible_row_count": int(len(eligible_df)),
        "quarantine_row_count": int(len(quarantine_df)),
        "quarantine_reason_counts": dict(
            Counter(r for q in quarantine_rows for r in q.get("quarantine_reasons", []))
        ),
        "class_counts_by_split_eligible": {k: dict(v) for k, v in class_by_split.items()},
        "family_split_isolation_eligible": {"ok": iso_ok, "errors": iso_errs[:20]},
        "cross_split_duplicate_code_hashes_eligible": cross_split_dups,
        "test_split_frozen_ref": "classification/test_split_frozen.json",
        "labels_unchanged_from_v3": True,
        "v3_files_unmodified": True,
    }

    metrics_dir = PROJECT_ROOT / "artifacts/metrics/phase13"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "benchmark_v4_build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if write_csv:
        clf = V4_ROOT / "classification"
        clf.mkdir(parents=True, exist_ok=True)
        if len(eligible_df):
            out_df = eligible_df.drop(columns=["_code_hash"], errors="ignore")
            out_df.to_csv(clf / "dataset.csv", index=False)
            manifest["eligible_csv_checksum_sha256"] = sha256_file(clf / "dataset.csv")
        else:
            (clf / "dataset.csv").write_text("", encoding="utf-8")
            manifest["eligible_csv_checksum_sha256"] = None
        if len(quarantine_df):
            qcols = [c for c in quarantine_df.columns if c != "_code_hash"]
            quarantine_df[qcols].to_csv(V4_ROOT / "quarantine.csv", index=False)
        (clf / "test_split_frozen.json").write_text(json.dumps(test_frozen, indent=2), encoding="utf-8")
        (V4_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def main() -> int:
    manifest = build_v4()
    print(json.dumps({"eligible": manifest["eligible_row_count"], "quarantine": manifest["quarantine_row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
