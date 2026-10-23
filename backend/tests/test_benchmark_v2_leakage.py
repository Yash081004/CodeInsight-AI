"""Leakage detection tests for benchmark_v2 pipeline."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_pipeline.splits import assign_groups, detect_pair_leakage, verify_group_isolation
from training.dataset_pipeline.similarity_v2 import code_hash, pair_key


class TestGroupIsolation:
    def test_detects_cross_split_groups(self):
        records = [
            {"family_id": "f1", "split": "train"},
            {"family_id": "f1", "split": "test"},
        ]
        ok, errs = verify_group_isolation(records, "family_id")
        assert ok is False
        assert len(errs) == 1

    def test_assign_groups_covers_all(self):
        groups = ["a", "b", "c", "d", "e", "f"]
        m = assign_groups(groups, seed=42)
        assert set(m.keys()) == set(groups)
        assert set(m.values()) <= {"train", "val", "test"}


class TestPairDedup:
    def test_reverse_order_same_key(self):
        a, b = "x = 1\n", "y = 2\n"
        assert pair_key(a, b) == pair_key(b, a)


class TestPairLeakage:
    def test_leakage_checker_catches_injected_mismatch(self):
        ca, cb = "x = 1\n", "y = 2\n"
        mapping = {code_hash(ca): "train", code_hash(cb): "test"}
        pairs = [{"pair_id": "PX", "code_a": ca, "code_b": cb, "split": "train"}]
        ok, errs = detect_pair_leakage(pairs, mapping, code_hash)
        assert ok is False
        assert errs

    def test_detects_mismatched_endpoint_splits(self):
        samples = {"h1": "train", "h2": "test"}
        pairs = [
            {
                "pair_id": "P1",
                "code_a": "code_train",
                "code_b": "code_test",
                "split": "train",
            }
        ]
        mapping = {
            code_hash("code_train"): "train",
            code_hash("code_test"): "test",
        }
        ok, errs = detect_pair_leakage(pairs, mapping, code_hash)
        assert ok is False

    def test_benchmark_v2_on_disk_if_present(self):
        root = PROJECT_ROOT / "data" / "benchmark_v2"
        csv = root / "classification" / "dataset.csv"
        if not csv.exists():
            pytest.skip("benchmark_v2 not generated")
        import pandas as pd

        df = pd.read_csv(csv)
        records = df.to_dict("records")
        ok, _ = verify_group_isolation(records, "family_id")
        assert ok is True
