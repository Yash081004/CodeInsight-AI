"""
Phase 12 comprehensive dataset readiness audit (read-only).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS
from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file
from training.phase12_template_tests import run_template_tests_on_code
from training.review_workflow import template_spec
from training.spec_catalog import load_catalog

DATASETS = {
    "legacy_v1": PROJECT_ROOT / "data/classification/dataset.csv",
    "benchmark_v2": PROJECT_ROOT / "data/benchmark_v2/classification/dataset.csv",
    "benchmark_v3": PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv",
}


def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def _hash(code: str) -> str:
    return hashlib.sha256(_norm(code).encode()).hexdigest()


def _ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def audit_v3_deep(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {"dataset": "benchmark_v3", "row_count": len(df)}
    report["checksum_sha256"] = sha256_file(PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv")

    for col in ("code", "error_label", "split", "family_id", "source_template_id", "problem_type", "expected_behavior"):
        if col in df.columns:
            empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
            report[f"empty_{col}_count"] = int(empty.sum())

    report["class_by_split"] = {
        sp: df[df["split"] == sp]["error_label"].value_counts().to_dict()
        for sp in sorted(df["split"].unique())
    }
    report["families_by_split"] = {
        sp: int(df[df["split"] == sp]["family_id"].nunique()) for sp in sorted(df["split"].unique())
    }
    report["templates_by_split"] = {
        sp: int(df[df["split"] == sp]["source_template_id"].nunique()) for sp in sorted(df["split"].unique())
    }

    ok, errs = verify_group_isolation(df.to_dict("records"), "family_id")
    report["family_id_split_isolation"] = {"ok": ok, "errors": errs[:10]}

    hashes = df["code"].map(_hash)
    dup = hashes[hashes.duplicated(keep=False)]
    cross_split_dups: List[Dict] = []
    if len(dup):
        for h in dup.unique():
            rows = df[hashes == h]
            splits = rows["split"].unique().tolist()
            if len(splits) > 1:
                cross_split_dups.append(
                    {
                        "code_hash": h,
                        "splits": splits,
                        "sample_ids": rows["sample_id"].tolist(),
                        "labels": rows["error_label"].tolist(),
                    }
                )
    report["cross_split_duplicate_code_hashes"] = cross_split_dups[:30]

    label_conflicts: List[Dict] = []
    for h in hashes.unique():
        rows = df[hashes == h]
        labels = rows["error_label"].unique()
        if len(labels) > 1:
            label_conflicts.append({"code_hash": h, "labels": labels.tolist(), "sample_ids": rows["sample_id"].tolist()})
    report["identical_code_different_label"] = label_conflicts

    near_dup_conflicts: List[Dict] = []
    codes = df["code"].tolist()
    ids = df["sample_id"].tolist()
    labels = df["error_label"].tolist()
    n = len(codes)
    checked = 0
    for i in range(min(n, 120)):
        for j in range(i + 1, min(n, 120)):
            checked += 1
            if labels[i] == labels[j]:
                continue
            r = SequenceMatcher(None, _norm(codes[i])[:500], _norm(codes[j])[:500]).ratio()
            if r >= 0.95:
                near_dup_conflicts.append(
                    {
                        "sample_a": ids[i],
                        "sample_b": ids[j],
                        "labels": [labels[i], labels[j]],
                        "similarity": round(r, 3),
                    }
                )
    report["near_duplicate_label_conflicts_sample"] = near_dup_conflicts[:25]
    report["near_duplicate_scan_pairs_checked"] = checked

    catalog = load_catalog()
    families = df["family_id"].dropna().unique()
    family_spec: Dict[str, Any] = {}
    for fam in sorted(families):
        ent = next((e for e in catalog.get("entries") or [] if e.get("family_id") == fam), None)
        templates = df[df["family_id"] == fam]["source_template_id"].unique().tolist()
        has_registry = any(template_spec(t).get("problem_statement") for t in templates)
        family_spec[str(fam)] = {
            "has_catalog_entry": ent is not None,
            "catalog_review_status": (ent or {}).get("provenance", {}).get("review_status"),
            "has_partial_registry_spec": has_registry,
            "trustworthy_nl_spec": (ent or {}).get("provenance", {}).get("review_status") == "human_reviewed",
        }
    report["family_specification_trust"] = family_spec
    report["families_without_human_reviewed_spec"] = [
        k for k, v in family_spec.items() if not v["trustworthy_nl_spec"]
    ]

    test_distinguish: List[Dict] = []
    for tid in df["source_template_id"].unique()[:40]:
        sub = df[df["source_template_id"] == tid]
        correct = sub[sub["error_label"] == "correct"]
        faults = sub[sub["error_label"] != "correct"]
        if correct.empty or faults.empty:
            continue
        c_row = correct.iloc[0]
        for _, f_row in faults.head(3).iterrows():
            c_res = run_template_tests_on_code(tid, c_row["code"])
            f_res = run_template_tests_on_code(tid, f_row["code"])
            test_distinguish.append(
                {
                    "template_id": tid,
                    "correct_sample": c_row["sample_id"],
                    "fault_sample": f_row["sample_id"],
                    "fault_label": f_row["error_label"],
                    "correct_passes_template_tests": c_res.get("passed_all"),
                    "fault_passes_template_tests": f_res.get("passed_all"),
                    "tests_distinguish_fault": c_res.get("passed_all") and not f_res.get("passed_all"),
                    "review_candidate_both_pass": c_res.get("passed_all") and f_res.get("passed_all"),
                }
            )
    report["template_test_distinguishability_sample"] = test_distinguish[:50]
    report["review_candidates_both_pass_count"] = sum(
        1 for x in test_distinguish if x.get("review_candidate_both_pass")
    )

    return report


def run_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 12,
        "datasets": {},
        "benchmark_v3_deep": None,
    }
    for name, path in DATASETS.items():
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        out["datasets"][name] = {
            "checksum": sha256_file(path),
            "rows": len(df),
            "class_counts": df["error_label"].value_counts().to_dict(),
        }
    v3 = pd.read_csv(DATASETS["benchmark_v3"])
    out["benchmark_v3_deep"] = audit_v3_deep(v3)
    return out


def main() -> int:
    out_dir = PROJECT_ROOT / "artifacts/metrics/phase12"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run_all()
    (out_dir / "dataset_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'dataset_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
