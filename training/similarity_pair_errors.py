"""
Per-pair similarity error artifact for benchmark_v2 TF-IDF evaluation.

Usage:
    py -3.13 -m training.similarity_pair_errors --run-id 20261019_062841
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_pipeline import DATASET_VERSION
from training.similarity_evaluation import (
    find_best_threshold,
    load_similarity_pairs,
    metrics_at_threshold,
    split_frame,
    tfidf_pair_similarities,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20261019_062841")
    args = parser.parse_args()

    pairs_path = PROJECT_ROOT / "data" / DATASET_VERSION / "similarity" / "pairs.csv"
    run_dir = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / args.run_id
    vec_path = PROJECT_ROOT / "models" / "benchmark_v2" / args.run_id / "similarity_tfidf_vectorizer.joblib"
    if not vec_path.exists():
        vec_path = run_dir / "similarity_tfidf_vectorizer.joblib"
    if not vec_path.exists():
        print(f"Missing vectorizer for run {args.run_id}")
        return 1

    df = load_similarity_pairs(pairs_path)
    val_df, test_df = split_frame(df)
    vec = joblib.load(vec_path)
    val_sims = tfidf_pair_similarities(vec, val_df["code_a"].tolist(), val_df["code_b"].tolist())
    test_sims = tfidf_pair_similarities(vec, test_df["code_a"].tolist(), test_df["code_b"].tolist())
    threshold, val_f1 = find_best_threshold(val_sims, val_df["label"].tolist())

    records = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        score = test_sims[i]
        pred = 1 if score >= threshold else 0
        true = int(row["label"])
        records.append(
            {
                "pair_id": row.get("pair_id", f"T{i}"),
                "family_a": row.get("family_a", ""),
                "family_b": row.get("family_b", ""),
                "pair_type": row.get("pair_type", ""),
                "true_label": true,
                "predicted_label": pred,
                "correct": pred == true,
                "similarity_score": round(float(score), 4),
                "threshold_val": threshold,
                "relation_note": "Label reflects typed pair relation — not universal semantic equivalence.",
            }
        )

    payload = {
        "dataset_version": DATASET_VERSION,
        "method": "tfidf_cosine",
        "test_pair_count": len(records),
        "validation_threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "threshold_tuned_on_test": False,
        "behavioral_test_limitation": "Finite template tests; passing does not prove equivalence.",
        "errors": [r for r in records if not r["correct"]],
        "pairs": records,
    }
    out_path = run_dir / "similarity_pair_errors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] {out_path} ({len(payload['errors'])} test errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
