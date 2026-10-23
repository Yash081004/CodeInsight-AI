"""
Pre-flight checks for CodeBERT classifier training.

Usage:
    py -3.13 -m training.verify_classifier_training
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID
from training.dataset_validation import sha256_file

EXPECTED_CHECKSUM = "c7ae7a8d836b64c9aa139e9c1fe77724329840a5629f87a08b3a2d1e993aff37"
EXPECTED_SPLITS = {"train": 61, "val": 14, "test": 16}


def verify(data_path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not data_path.exists():
        return False, [f"Missing dataset: {data_path}"]

    checksum = sha256_file(data_path)
    if checksum != EXPECTED_CHECKSUM:
        errors.append(f"Dataset checksum mismatch: got {checksum}")

    df = pd.read_csv(data_path)
    if len(df) != 91:
        errors.append(f"Expected 91 samples, got {len(df)}")

    invalid = set(df["error_label"]) - set(ERROR_LABELS)
    if invalid:
        errors.append(f"Invalid labels: {invalid}")

    if LABEL_TO_ID != {
        "correct": 0,
        "wrong_initialization": 1,
        "off_by_one_loop": 2,
        "wrong_operator": 3,
        "wrong_condition": 4,
    }:
        errors.append("LABEL_TO_ID mapping drift from config")

    for idx, label in ID_TO_LABEL.items():
        if LABEL_TO_ID[label] != idx:
            errors.append(f"Non-bijective label map at id {idx}")

    for split, n in EXPECTED_SPLITS.items():
        got = int((df["split"] == split).sum())
        if got != n:
            errors.append(f"Split {split}: expected {n}, got {got}")

    return len(errors) == 0, errors


def main() -> int:
    data_path = PROJECT_ROOT / "data" / "classification" / "dataset.csv"
    ok, errors = verify(data_path)
    if ok:
        print("[OK] Training pre-flight checks passed.")
        print(f"  Dataset checksum: {EXPECTED_CHECKSUM[:8]}...{EXPECTED_CHECKSUM[-8:]}")
        print(f"  Splits: train=61, val=14, test=16 (test held out until final eval)")
        return 0
    print("[FAIL] Pre-flight checks failed:")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
