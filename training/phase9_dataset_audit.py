"""
Phase 9: read-only dataset and split audit (no label or checkpoint changes).

Usage:
  py -3.13 -m training.phase9_dataset_audit
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
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS
from training.dataset_validation import sha256_file

DATASETS = {
    "legacy_v1": PROJECT_ROOT / "data/classification/dataset.csv",
    "benchmark_v2": PROJECT_ROOT / "data/benchmark_v2/classification/dataset.csv",
    "benchmark_v3": PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv",
}

MANIFESTS = {
    "benchmark_v2": PROJECT_ROOT / "data/benchmark_v2/manifest.json",
    "benchmark_v3": PROJECT_ROOT / "data/benchmark_v3/manifest.json",
}

OUT_DIR = PROJECT_ROOT / "artifacts/metrics/phase9"


def _norm_code(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def _code_hash(code: str) -> str:
    return hashlib.sha256(_norm_code(code).encode("utf-8")).hexdigest()[:16]


def _syntax_ok(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def _family_key(row: pd.Series) -> str:
    """Split-isolation key: family_id (v2/v3) or full source_template_id (legacy v1)."""
    if "family_id" in row and pd.notna(row.get("family_id")):
        return str(row["family_id"])
    if "source_template_id" in row and pd.notna(row.get("source_template_id")):
        return str(row["source_template_id"])
    if "problem_type" in row and pd.notna(row.get("problem_type")):
        return str(row["problem_type"])
    return "unknown"


def audit_dataframe(name: str, path: Path, manifest_path: Optional[Path]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset_name": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.is_file(),
    }
    if not path.is_file():
        report["error"] = "missing"
        return report

    report["checksum_sha256"] = sha256_file(path)
    df = pd.read_csv(path)
    report["total_samples"] = len(df)

    label_col = "error_label"
    split_col = "split"
    invalid_labels = set(df[label_col].dropna().unique()) - set(ERROR_LABELS)
    report["invalid_label_values"] = sorted(invalid_labels)
    report["class_counts"] = df[label_col].value_counts().to_dict()
    report["split_counts"] = df[split_col].value_counts().to_dict() if split_col in df.columns else {}

    by_split_label: Dict[str, Dict[str, int]] = {}
    if split_col in df.columns:
        for sp in sorted(df[split_col].dropna().unique()):
            sub = df[df[split_col] == sp]
            by_split_label[str(sp)] = sub[label_col].value_counts().to_dict()
    report["class_counts_per_split"] = by_split_label

    families = df.apply(_family_key, axis=1)
    report["independent_families"] = int(families.nunique())
    if "problem_type" in df.columns:
        pt = df["problem_type"].dropna()
        report["independent_problem_types"] = int(pt.nunique())
        pt_by_split: Dict[str, Set[str]] = defaultdict(set)
        if split_col in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row.get("problem_type")):
                    pt_by_split[str(row[split_col])].add(str(row["problem_type"]))
        cross_pt = []
        sp = list(pt_by_split.keys())
        for i, a in enumerate(sp):
            for b in sp[i + 1 :]:
                inter = pt_by_split[a] & pt_by_split[b]
                if inter:
                    cross_pt.append({"splits": [a, b], "problem_types": sorted(inter)})
        report["problem_type_overlap_across_splits"] = cross_pt
    fam_by_split: Dict[str, Set[str]] = defaultdict(set)
    if split_col in df.columns:
        for _, row in df.iterrows():
            fam_by_split[str(row[split_col])].add(_family_key(row))
    report["families_per_split"] = {k: sorted(v) for k, v in fam_by_split.items()}
    report["families_per_split_counts"] = {k: len(v) for k, v in fam_by_split.items()}

    overlap: Dict[str, Any] = {"template_leakage": False, "family_leakage": False, "overlapping_families": []}
    if len(fam_by_split) >= 2:
        splits = list(fam_by_split.keys())
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                inter = fam_by_split[a] & fam_by_split[b]
                if inter:
                    overlap["family_leakage"] = True
                    overlap["overlapping_families"].append({"splits": [a, b], "families": sorted(inter)})
    report["split_overlap"] = overlap

    if "source_template_id" in df.columns and split_col in df.columns:
        tmpl_splits: Dict[str, Set[str]] = defaultdict(set)
        for _, row in df.iterrows():
            tmpl_splits[str(row["source_template_id"])].add(str(row[split_col]))
        leaked = {t: sorted(s) for t, s in tmpl_splits.items() if len(s) > 1}
        report["template_id_split_leakage"] = leaked
        if leaked:
            overlap["template_leakage"] = True

    dup_ids = df["sample_id"].duplicated().sum() if "sample_id" in df.columns else 0
    report["duplicate_sample_ids"] = int(dup_ids)

    hashes = df["code"].map(_code_hash)
    dup_body = hashes.duplicated().sum()
    report["duplicate_normalized_code_bodies"] = int(dup_body)
    if dup_body:
        dup_groups = (
            df.assign(_h=hashes)
            .groupby("_h")["sample_id"]
            .apply(lambda s: s.tolist())
            .to_dict()
        )
        report["duplicate_code_groups_sample"] = {
            k: v for k, v in list(dup_groups.items())[:10] if len(v) > 1
        }

    near_dup_pairs: List[Dict[str, Any]] = []
    codes = df["code"].fillna("").tolist()
    ids = df["sample_id"].tolist() if "sample_id" in df.columns else list(range(len(df)))
    labels = df[label_col].tolist()
    # O(n^2) only for small legacy; cap for v3
    max_pairs = 8000
    checked = 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if labels[i] != labels[j]:
                continue
            checked += 1
            if checked > max_pairs:
                break
            ratio = SequenceMatcher(None, _norm_code(codes[i])[:400], _norm_code(codes[j])[:400]).ratio()
            if ratio >= 0.92 and _code_hash(codes[i]) != _code_hash(codes[j]):
                near_dup_pairs.append(
                    {
                        "sample_a": ids[i],
                        "sample_b": ids[j],
                        "label": labels[i],
                        "similarity": round(ratio, 3),
                    }
                )
        if checked > max_pairs:
            break
    report["near_duplicate_same_label_pairs"] = near_dup_pairs[:25]
    report["near_duplicate_scan_capped"] = checked >= max_pairs

    syntax_bad: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        ok, err = _syntax_ok(str(row.get("code", "")))
        if not ok:
            syntax_bad.append({"sample_id": str(row.get("sample_id")), "error": err or "syntax"})
    report["syntax_invalid_samples"] = syntax_bad

    spec_field = "expected_behavior" if "expected_behavior" in df.columns else None
    empty_spec = 0
    if spec_field:
        empty_spec = int(df[spec_field].isna().sum() + (df[spec_field].astype(str).str.strip() == "").sum())
        report["empty_expected_behavior_count"] = empty_spec

    concerns: List[Dict[str, Any]] = []
    verified: List[Dict[str, Any]] = []

    if "validation_status" in df.columns:
        flagged = df[df["validation_status"].astype(str).str.lower() != "valid"]
        for _, row in flagged.iterrows():
            concerns.append(
                {
                    "type": "validation_status_not_valid",
                    "sample_id": row.get("sample_id"),
                    "status": row.get("validation_status"),
                    "label": row.get(label_col),
                    "confidence": "verified",
                }
            )

    if manifest_path and manifest_path.is_file():
        report["manifest_checksum"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("checksum_sha256")

    rq_v3 = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
    if name == "benchmark_v3" and rq_v3.is_file():
        rq = json.loads(rq_v3.read_text(encoding="utf-8"))
        report["review_queue_count"] = len(rq)
        report["review_queue_reasons"] = Counter(x.get("reason") for x in rq).most_common()
        for item in rq:
            concerns.append(
                {
                    "type": "mutation_still_passes_tests",
                    "template_id": item.get("template_id"),
                    "intended_label": item.get("intended_label"),
                    "confidence": "verified",
                    "note": "Synthetic label may not match observable behavior under bundled tests.",
                }
            )

    if overlap["family_leakage"]:
        concerns.append(
            {
                "type": "family_split_leakage",
                "detail": overlap["overlapping_families"],
                "confidence": "verified",
            }
        )
    else:
        verified.append({"type": "no_family_id_overlap_across_splits", "confidence": "verified"})

    if not invalid_labels and not syntax_bad:
        verified.append({"type": "labels_and_syntax_schema_ok", "confidence": "verified"})

    report["verified_findings"] = verified
    report["concerns"] = concerns
    return report


def load_metric_snapshot() -> Dict[str, Any]:
    """Reference existing evaluation artifacts (no retraining)."""
    root = PROJECT_ROOT / "artifacts/metrics"

    def _load(rel: str) -> Optional[Dict]:
        p = root / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    return {
        "legacy_v1_test": {
            "codebert": _load("codebert_classification_metrics.json"),
            "tfidf": _load("baseline_classification_metrics.json"),
            "confusion_codebert": _load("confusion_matrix.json"),
        },
        "benchmark_v2_test": {
            "codebert": _load("benchmark_v2/codebert_20261019_082413/codebert_classification_metrics.json"),
            "tfidf": _load("benchmark_v2/20261019_062841/classification_metrics.json"),
        },
        "similarity_v1_test": _load("similarity_hybrid_policy.json"),
        "production_training_config": (
            json.loads((PROJECT_ROOT / "models/codebert_classifier/training_config.json").read_text(encoding="utf-8"))
            if (PROJECT_ROOT / "models/codebert_classifier/training_config.json").is_file()
            else None
        ),
        "holdout_contamination_notes": {
            "legacy_v1_test": "Used for production checkpoint selection (best_val_f1) and repeated QA — contaminated for fresh model selection.",
            "benchmark_v2_test": "Documented in collapse_audit.json as previously inspected.",
            "benchmark_v3_test": "Preferred forward holdout if not used for tuning; verify experiment logs before selection.",
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    datasets = {}
    for name, path in DATASETS.items():
        manifest = MANIFESTS.get(name)
        datasets[name] = audit_dataframe(name, path, manifest)

    payload = {
        "generated_at": ts,
        "datasets": datasets,
        "metrics_snapshot": load_metric_snapshot(),
        "label_definitions": ERROR_LABELS,
    }

    out_json = OUT_DIR / "dataset_audit_report.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
