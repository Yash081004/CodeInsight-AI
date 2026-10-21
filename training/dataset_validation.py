"""
Dataset validation for CodeInsight AI classification and similarity corpora.

Flags invalid records; does not silently relabel.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backend.app.config import ERROR_LABELS
from training.generate_dataset import TEMPLATES, run_test_safely, validate_syntax

REQUIRED_CLASSIFICATION_FIELDS = [
    "sample_id",
    "source_template_id",
    "problem_type",
    "code",
    "error_label",
    "mutation_type",
    "mutation_description",
    "expected_behavior",
    "validation_status",
    "split",
]

VALID_SPLITS = {"train", "val", "test"}
TEMPLATE_BY_ID = {t["template_id"]: t for t in TEMPLATES}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _pair_key(code_a: str, code_b: str) -> Tuple[str, str]:
    ha, hb = _code_hash(code_a), _code_hash(code_b)
    return (ha, hb) if ha <= hb else (hb, ha)


def validate_classification_dataset(
    dataset_path: Path,
    *,
    run_behavioral_checks: bool = True,
) -> Dict[str, Any]:
    """Validate classification CSV; return structured report with issues list."""
    issues: List[Dict[str, str]] = []
    flagged_sample_ids: Set[str] = set()

    if not dataset_path.exists():
        return {
            "valid": False,
            "error": f"Missing dataset: {dataset_path}",
            "issues": [{"severity": "error", "message": "dataset file not found"}],
        }

    with open(dataset_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sample_ids: List[str] = []
    code_hashes: Dict[str, str] = {}
    template_splits: Dict[str, str] = {}
    split_counts: Counter = Counter()
    class_counts: Counter = Counter()
    class_by_split: Dict[str, Counter] = defaultdict(Counter)
    invalid_label_count = 0
    syntax_invalid = 0
    behavioral_mismatch = 0
    missing_field_count = 0
    unknown_template_count = 0

    for row in rows:
        sid = row.get("sample_id", "")
        sample_ids.append(sid)

        for field in REQUIRED_CLASSIFICATION_FIELDS:
            if field not in row or row[field] is None or str(row[field]).strip() == "":
                missing_field_count += 1
                issues.append(
                    {
                        "severity": "error",
                        "sample_id": sid,
                        "message": f"Missing required field: {field}",
                    }
                )
                flagged_sample_ids.add(sid)

        label = row.get("error_label", "")
        if label not in ERROR_LABELS:
            invalid_label_count += 1
            issues.append(
                {
                    "severity": "error",
                    "sample_id": sid,
                    "message": f"Invalid error_label: {label!r}",
                }
            )
            flagged_sample_ids.add(sid)
        else:
            class_counts[label] += 1

        split = row.get("split", "")
        if split not in VALID_SPLITS:
            issues.append(
                {
                    "severity": "error",
                    "sample_id": sid,
                    "message": f"Invalid split: {split!r}",
                }
            )
            flagged_sample_ids.add(sid)
        else:
            split_counts[split] += 1
            if label in ERROR_LABELS:
                class_by_split[split][label] += 1

        code = row.get("code", "")
        if not validate_syntax(code):
            syntax_invalid += 1
            issues.append(
                {
                    "severity": "error",
                    "sample_id": sid,
                    "message": "AST syntax validation failed",
                }
            )
            flagged_sample_ids.add(sid)

        ch = _code_hash(code)
        if ch in code_hashes and code_hashes[ch] != sid:
            issues.append(
                {
                    "severity": "warning",
                    "sample_id": sid,
                    "message": f"Duplicate code body (same as {code_hashes[ch]})",
                }
            )
        code_hashes[ch] = sid

        tid = row.get("source_template_id", "")
        if tid not in TEMPLATE_BY_ID:
            unknown_template_count += 1
            issues.append(
                {
                    "severity": "warning",
                    "sample_id": sid,
                    "message": f"Unknown source_template_id: {tid}",
                }
            )
        else:
            if tid in template_splits and template_splits[tid] != split:
                issues.append(
                    {
                        "severity": "error",
                        "sample_id": sid,
                        "message": (
                            f"Template {tid} spans splits "
                            f"({template_splits[tid]} vs {split}) — leakage"
                        ),
                    }
                )
                flagged_sample_ids.add(sid)
            else:
                template_splits[tid] = split

        if run_behavioral_checks and tid in TEMPLATE_BY_ID and validate_syntax(code):
            test_code = TEMPLATE_BY_ID[tid]["test_code"]
            passed, err = run_test_safely(code, test_code)
            expected_pass = label == "correct"
            if passed != expected_pass:
                behavioral_mismatch += 1
                issues.append(
                    {
                        "severity": "error",
                        "sample_id": sid,
                        "message": (
                            f"Behavioral check mismatch: tests_passed={passed}, "
                            f"label={label!r}; detail={err[:200]}"
                        ),
                    }
                )
                flagged_sample_ids.add(sid)

    dup_ids = [sid for sid, c in Counter(sample_ids).items() if c > 1]

    unique_templates = len(template_splits)
    stratification_note = (
        "Splits are grouped by source_template_id (seed=42 in generator). "
        "Per-class stratification is not applied because all variants of a template "
        "share one split; with 91 samples and 5 classes, grouped stratification "
        "is not statistically meaningful."
    )

    report: Dict[str, Any] = {
        "valid": len(flagged_sample_ids) == 0 and not dup_ids and invalid_label_count == 0,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "checksum_sha256": sha256_file(dataset_path),
        "total_samples": len(rows),
        "unique_source_templates": unique_templates,
        "class_distribution": dict(class_counts),
        "samples_per_split": dict(split_counts),
        "class_counts_per_split": {k: dict(v) for k, v in class_by_split.items()},
        "duplicate_sample_ids": dup_ids,
        "duplicate_code_bodies": sum(1 for i in issues if "Duplicate code" in i.get("message", "")),
        "invalid_label_count": invalid_label_count,
        "syntax_invalid_count": syntax_invalid,
        "behavioral_mismatch_count": behavioral_mismatch,
        "missing_field_violations": missing_field_count,
        "unknown_template_count": unknown_template_count,
        "flagged_invalid_sample_count": len(flagged_sample_ids),
        "split_method": "grouped_by_source_template_id",
        "split_seed": 42,
        "stratification_limitation": stratification_note,
        "template_split_leakage_checked": True,
        "template_split_leakage_detected": any(
            "spans splits" in i.get("message", "") for i in issues
        ),
        "issues": issues,
    }
    return report


def validate_similarity_pairs(
    pairs_path: Path,
    classification_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate pairs.csv; optional cross-check against classification splits via code hash."""
    issues: List[Dict[str, str]] = []

    if not pairs_path.exists():
        return {
            "valid": False,
            "error": f"Missing pairs file: {pairs_path}",
            "issues": [{"severity": "error", "message": "pairs.csv not found"}],
        }

    df = pd.read_csv(pairs_path)
    code_to_split: Dict[str, str] = {}
    if classification_path and classification_path.exists():
        clf = pd.read_csv(classification_path)
        for _, row in clf.iterrows():
            code_to_split[_code_hash(str(row["code"]))] = str(row["split"])

    pair_keys: Dict[Tuple[str, str], str] = {}
    cross_split_rows = 0
    endpoint_mismatch = 0
    missing_code = 0
    invalid_label = 0

    split_counts = Counter()
    label_counts = Counter()

    for _, row in df.iterrows():
        pid = str(row.get("pair_id", ""))
        split = str(row.get("split", ""))
        split_counts[split] += 1
        if split == "cross_split":
            cross_split_rows += 1

        label = row.get("label")
        try:
            label_int = int(label)
            if label_int not in (0, 1):
                raise ValueError("label not binary")
            label_counts[label_int] += 1
        except (TypeError, ValueError):
            invalid_label += 1
            issues.append({"severity": "error", "pair_id": pid, "message": f"Invalid label: {label!r}"})

        ca, cb = str(row.get("code_a", "")), str(row.get("code_b", ""))
        if not ca.strip() or not cb.strip():
            missing_code += 1
            issues.append({"severity": "error", "pair_id": pid, "message": "Empty code_a or code_b"})
            continue
        if not validate_syntax(ca) or not validate_syntax(cb):
            issues.append({"severity": "error", "pair_id": pid, "message": "Syntax invalid in pair snippet"})

        pk = _pair_key(ca, cb)
        if pk in pair_keys:
            issues.append(
                {
                    "severity": "warning",
                    "pair_id": pid,
                    "message": f"Duplicate pair (same as {pair_keys[pk]})",
                }
            )
        pair_keys[pk] = pid

        if code_to_split:
            sa = code_to_split.get(_code_hash(ca))
            sb = code_to_split.get(_code_hash(cb))
            if sa and sb:
                if sa != sb:
                    if split != "cross_split":
                        endpoint_mismatch += 1
                        issues.append(
                            {
                                "severity": "error",
                                "pair_id": pid,
                                "message": (
                                    f"Endpoint splits differ ({sa} vs {sb}) "
                                    f"but pair tagged {split!r} (expected cross_split)"
                                ),
                            }
                        )
                elif split in VALID_SPLITS and sa != split:
                    endpoint_mismatch += 1
                    issues.append(
                        {
                            "severity": "error",
                            "pair_id": pid,
                            "message": f"Pair tagged {split} but endpoints are both {sa}",
                        }
                    )

    evaluable = df[df["split"].isin(VALID_SPLITS)] if "split" in df.columns else df

    report: Dict[str, Any] = {
        "valid": endpoint_mismatch == 0 and invalid_label == 0 and missing_code == 0,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "pairs_path": str(pairs_path),
        "checksum_sha256": sha256_file(pairs_path),
        "total_pairs": len(df),
        "cross_split_pairs": cross_split_rows,
        "evaluable_pairs_excluding_cross_split": len(evaluable),
        "label_distribution": {"similar": int(label_counts.get(1, 0)), "dissimilar": int(label_counts.get(0, 0))},
        "samples_per_split": dict(split_counts),
        "duplicate_pair_count": sum(1 for i in issues if "Duplicate pair" in i.get("message", "")),
        "endpoint_split_mismatch_count": endpoint_mismatch,
        "pair_leakage_check_method": (
            "pair.split must equal both endpoint splits when codes match classification corpus"
        ),
        "issues": issues,
    }
    return report


def save_dataset_metadata(
    classification_report: Dict[str, Any],
    similarity_report: Dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": classification_report.get("checksum_sha256", "")[:12],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification_report,
        "similarity": similarity_report,
    }
    path = output_dir / "dataset_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path
