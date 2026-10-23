"""Phase 10 review-queue exclusion logic."""

import json
from pathlib import Path

import pandas as pd

from training.phase10_v3_exclusions import (
    apply_train_val_template_exclusion,
    load_review_queue_template_ids,
    verify_no_family_leakage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V3 = PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv"
QUEUE = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"


def test_exclusion_does_not_touch_test_split():
    df = pd.read_csv(V3)
    tids = load_review_queue_template_ids(QUEUE)
    filtered, excluded = apply_train_val_template_exclusion(df, tids)
    assert len(excluded) > 0
    assert (excluded["split"] != "test").all()
    test_before = set(df[df["split"] == "test"]["sample_id"])
    test_after = set(filtered[filtered["split"] == "test"]["sample_id"])
    assert test_before == test_after
    ok, errs = verify_no_family_leakage(filtered)
    assert ok, errs
