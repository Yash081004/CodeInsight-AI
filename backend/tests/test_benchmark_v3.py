import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_pipeline.splits_v3 import (
    refine_splits_for_label_coverage,
    class_coverage_by_split,
)
from training.label_review_cli import _load_decisions, VALID_STATUS


def test_refine_splits_preserves_families():
    family_labels = {
        "f1": {"correct", "off_by_one_loop"},
        "f2": {"correct", "wrong_operator"},
        "f3": {"correct", "wrong_condition"},
        "f4": {"correct", "wrong_initialization"},
        "f5": {"correct", "off_by_one_loop"},
        "f6": {"correct", "wrong_operator"},
    }
    mapping = {f: "train" for f in family_labels}
    refined, _ = refine_splits_for_label_coverage(mapping, family_labels, seed=42)
    assert set(refined.keys()) == set(family_labels.keys())
    assert all(sp in ("train", "val", "test") for sp in refined.values())


def test_review_decisions_all_pending_initially():
    data = _load_decisions()
    assert data["human_reviewed"] is False
    for d in data["decisions"]:
        assert d["status"] in VALID_STATUS
        assert d["status"] == "pending"


def test_benchmark_v3_dataset_if_present():
    csv = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    if not csv.exists():
        pytest.skip("benchmark_v3 not generated")
    import pandas as pd
    from training.dataset_pipeline.splits import verify_group_isolation

    df = pd.read_csv(csv)
    ok, _ = verify_group_isolation(df.to_dict("records"), "family_id")
    assert ok is True
    cov = class_coverage_by_split(df.to_dict("records"))
    assert "train" in cov


def test_v2_checksum_unchanged():
    csv = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"
    if not csv.exists():
        pytest.skip("v2 missing")
    from training.dataset_validation import sha256_file

    assert sha256_file(csv) == "6d2d6ab93918ef86ebbb36d46b0ced5f109dbc0983f2229a067fe72ef194ba55"
