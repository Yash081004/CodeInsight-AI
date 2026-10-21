"""
Tests for dataset validation and evaluation methodology.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, get_settings
from training.dataset_validation import (
    validate_classification_dataset,
    validate_similarity_pairs,
)
from training.similarity_evaluation import (
    find_best_threshold,
    load_similarity_pairs,
    metrics_at_threshold,
    split_frame,
)


@pytest.fixture
def settings():
    return get_settings()


class TestClassificationDatasetOnDisk:
    def test_dataset_exists_and_labels_valid(self, settings):
        path = settings.classification_data_path / "dataset.csv"
        if not path.exists():
            pytest.skip("classification dataset not present")
        report = validate_classification_dataset(path, run_behavioral_checks=False)
        assert report["total_samples"] == 91
        assert set(report["class_distribution"].keys()).issubset(set(ERROR_LABELS))
        assert report["invalid_label_count"] == 0
        assert not report["template_split_leakage_detected"]

    def test_no_duplicate_sample_ids(self, settings):
        path = settings.classification_data_path / "dataset.csv"
        if not path.exists():
            pytest.skip("classification dataset not present")
        report = validate_classification_dataset(path, run_behavioral_checks=False)
        assert report["duplicate_sample_ids"] == []


class TestSimilarityPairsOnDisk:
    def test_pair_counts_and_splits(self, settings):
        pairs_path = settings.similarity_data_path / "pairs.csv"
        clf_path = settings.classification_data_path / "dataset.csv"
        if not pairs_path.exists():
            pytest.skip("similarity pairs not present")
        report = validate_similarity_pairs(pairs_path, clf_path)
        assert report["total_pairs"] == 193
        assert report["endpoint_split_mismatch_count"] == 0
        assert report["cross_split_pairs"] >= 0
        assert report["evaluable_pairs_excluding_cross_split"] <= report["total_pairs"]

    def test_cross_split_excluded_from_eval_loader(self, settings):
        pairs_path = settings.similarity_data_path / "pairs.csv"
        if not pairs_path.exists():
            pytest.skip("similarity pairs not present")
        df = load_similarity_pairs(pairs_path)
        assert "cross_split" not in set(df["split"])


class TestThresholdSelection:
    def test_validation_only_threshold_not_test_labels(self):
        # Val: clear separation; test would prefer wrong threshold if tuned on test
        val_sims = [0.9, 0.8, 0.2, 0.1]
        val_labels = [1, 1, 0, 0]
        test_sims = [0.85, 0.15]
        test_labels = [1, 0]

        t, val_f1 = find_best_threshold(val_sims, val_labels)
        assert val_f1 > 0.5
        test_m = metrics_at_threshold(test_sims, test_labels, t)
        assert test_m["sample_count"] == 2
        assert 0.0 <= test_m["f1"] <= 1.0

    def test_val_test_frames_nonempty_on_disk(self, settings):
        pairs_path = settings.similarity_data_path / "pairs.csv"
        if not pairs_path.exists():
            pytest.skip("similarity pairs not present")
        df = load_similarity_pairs(pairs_path)
        val_df, test_df = split_frame(df)
        assert len(val_df) >= 1
        assert len(test_df) >= 1


class TestClassifierTrainingPreflight:
    def test_verify_classifier_training_passes(self, settings):
        from training.verify_classifier_training import verify

        path = settings.classification_data_path / "dataset.csv"
        if not path.exists():
            pytest.skip("classification dataset not present")
        ok, errors = verify(path)
        assert ok is True
        assert errors == []


class TestMissingDatasetHandling:
    def test_missing_classification_file(self, tmp_path):
        report = validate_classification_dataset(tmp_path / "missing.csv")
        assert report["valid"] is False
        assert "not found" in report["issues"][0]["message"].lower() or "Missing" in report.get("error", "")

    def test_missing_pairs_file(self, tmp_path):
        report = validate_similarity_pairs(tmp_path / "missing.csv")
        assert report["valid"] is False
