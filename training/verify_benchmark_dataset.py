"""
Pre-flight checks for benchmark_v2 (or any versioned classification CSV with family_id).

Usage:
    py -3.13 -m training.verify_benchmark_dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID
from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file


def verify(data_path: Path, manifest_path: Path | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not data_path.exists():
        return False, [f"Missing dataset: {data_path}"]

    df = pd.read_csv(data_path)
    required = {"sample_id", "code", "error_label", "split", "family_id", "source_template_id"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    invalid = set(df["error_label"]) - set(ERROR_LABELS)
    if invalid:
        errors.append(f"Invalid labels: {invalid}")

    for split in ("train", "val", "test"):
        if split not in df["split"].values:
            errors.append(f"Missing split: {split}")

    if "family_id" in df.columns:
        ok, iso_errs = verify_group_isolation(df.to_dict("records"), "family_id")
        if not ok:
            errors.extend(iso_errs[:5])
            if len(iso_errs) > 5:
                errors.append(f"... and {len(iso_errs) - 5} more isolation errors")

    checksum = sha256_file(data_path)
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        expected = manifest.get("checksum_sha256")
        if expected and checksum != expected:
            errors.append(f"Checksum mismatch: file {checksum[:16]}... vs manifest {expected[:16]}...")

    families = df["family_id"].nunique() if "family_id" in df.columns else 0
    if families < 10:
        errors.append(f"Only {families} independent families — benchmark may be too small for strong claims")

    return len(errors) == 0, errors


def main() -> int:
    root = PROJECT_ROOT / "data" / "benchmark_v2"
    data_path = root / "classification" / "dataset.csv"
    ok, errors = verify(data_path, root / "manifest.json")
    if ok:
        df = pd.read_csv(data_path)
        print("[OK] benchmark_v2 pre-flight passed.")
        print(f"  samples={len(df)} families={df['family_id'].nunique()} checksum={sha256_file(data_path)[:16]}...")
        return 0
    print("[FAIL] benchmark_v2 pre-flight:")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
