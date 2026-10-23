"""
Phase 10: benchmark v3 review-queue exclusions (read-only on source CSV labels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file


def load_review_queue_template_ids(queue_path: Path) -> Set[str]:
    data = json.loads(queue_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {queue_path}")
    ids: Set[str] = set()
    for row in data:
        tid = row.get("template_id")
        if tid:
            ids.add(str(tid))
    return ids


def apply_train_val_template_exclusion(
    df: pd.DataFrame,
    template_ids: Set[str],
    template_column: str = "source_template_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop train/val rows whose source_template_id is in template_ids. Test split untouched."""
    if template_column not in df.columns:
        raise KeyError(f"Missing column {template_column}")
    in_queue = df[template_column].astype(str).isin(template_ids)
    train_val = df["split"].isin(["train", "val"])
    exclude_mask = in_queue & train_val
    excluded = df[exclude_mask].copy()
    kept = df[~exclude_mask].copy()
    return kept, excluded


def exclusion_manifest(
    full_checksum: str,
    queue_path: Path,
    template_ids: Set[str],
    excluded: pd.DataFrame,
    kept: pd.DataFrame,
) -> Dict[str, Any]:
    fam_col = "family_id" if "family_id" in excluded.columns else "problem_type"
    return {
        "source_dataset_checksum_sha256": full_checksum,
        "review_queue_path": str(queue_path),
        "excluded_template_ids": sorted(template_ids),
        "excluded_row_count": int(len(excluded)),
        "excluded_sample_ids": excluded["sample_id"].tolist() if "sample_id" in excluded.columns else [],
        "excluded_families": sorted(excluded[fam_col].dropna().unique().tolist()) if len(excluded) else [],
        "remaining_train_val_count": int(len(kept[kept["split"].isin(["train", "val"])])),
        "remaining_test_count": int(len(kept[kept["split"] == "test"])),
        "labels_unchanged": True,
    }


def verify_no_family_leakage(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    return verify_group_isolation(df.to_dict("records"), "family_id")


def write_filtered_view_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return sha256_file(path)
