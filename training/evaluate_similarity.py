"""
Evaluate TF-IDF and CodeBERT similarity on held-out test pairs.

Thresholds are selected on validation pairs only.

Usage:
    py -3.13 -m training.evaluate_similarity
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.models.codebert_encoder import CodeBERTEncoder
from training.evaluation_artifacts import save_binary_confusion
from training.similarity_evaluation import (
    build_similarity_metrics_payload,
    find_best_threshold,
    load_similarity_pairs,
    metrics_at_threshold,
    predictions_at_threshold,
    save_similarity_metrics,
    split_frame,
    tfidf_pair_similarities,
)


def codebert_pair_similarities(encoder: CodeBERTEncoder, codes_a, codes_b, max_length: int = 512):
    sims = []
    for ca, cb in zip(codes_a, codes_b):
        score = encoder.compute_similarity(ca, cb, max_length=max_length)
        sims.append(float(score) if score is not None else 0.0)
    return sims


def _write_method_report(
    metrics_dir: Path,
    prefix: str,
    test_metrics: dict,
    threshold: float,
    val_f1: float,
    y_true,
    y_pred,
) -> None:
    report = {
        "model": prefix,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "threshold_selection": "validation_max_f1_grid_0.05",
        "validation_threshold_f1": val_f1,
        "test_threshold": threshold,
        "test_sample_count": test_metrics.get("sample_count"),
        **{k: test_metrics.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    }
    with open(metrics_dir / f"{prefix}_similarity_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    save_binary_confusion(
        y_true,
        y_pred,
        metrics_dir / f"{prefix}_similarity_confusion.json",
    )


def main():
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"
    pairs_path = settings.similarity_data_path / "pairs.csv"
    if not pairs_path.exists():
        print(f"ERROR: Missing {pairs_path}")
        return 1

    df = load_similarity_pairs(pairs_path)
    val_df, test_df = split_frame(df)
    if len(val_df) == 0 or len(test_df) == 0:
        print("ERROR: Need non-empty val and test similarity splits.")
        return 1

    vec_path = settings.baselines_model_path / "similarity_tfidf_vectorizer.joblib"
    if not vec_path.exists():
        print(f"ERROR: Missing TF-IDF vectorizer at {vec_path}")
        print("Run: py -3.13 -m training.train_baselines")
        return 1

    vectorizer = joblib.load(vec_path)

    val_tfidf = tfidf_pair_similarities(
        vectorizer, val_df["code_a"].tolist(), val_df["code_b"].tolist()
    )
    test_tfidf = tfidf_pair_similarities(
        vectorizer, test_df["code_a"].tolist(), test_df["code_b"].tolist()
    )
    tfidf_threshold, tfidf_val_f1 = find_best_threshold(val_tfidf, val_df["label"].tolist())
    tfidf_test_metrics = metrics_at_threshold(
        test_tfidf, test_df["label"].tolist(), tfidf_threshold
    )
    tfidf_preds = predictions_at_threshold(test_tfidf, tfidf_threshold)

    print(f"TF-IDF threshold (val): {tfidf_threshold:.2f} (val F1={tfidf_val_f1:.4f})")
    print(
        f"TF-IDF test (n={len(test_df)}) — acc={tfidf_test_metrics['accuracy']:.4f} "
        f"f1={tfidf_test_metrics['f1']:.4f} "
        f"roc_auc={tfidf_test_metrics.get('roc_auc')}"
    )

    codebert_test_metrics = None
    codebert_threshold = None
    codebert_val_f1 = None

    encoder = CodeBERTEncoder(model_name=settings.codebert_model_name, device=settings.device)
    if encoder.load():
        val_cb = codebert_pair_similarities(
            encoder, val_df["code_a"].tolist(), val_df["code_b"].tolist()
        )
        test_cb = codebert_pair_similarities(
            encoder, test_df["code_a"].tolist(), test_df["code_b"].tolist()
        )
        codebert_threshold, codebert_val_f1 = find_best_threshold(val_cb, val_df["label"].tolist())
        codebert_test_metrics = metrics_at_threshold(
            test_cb, test_df["label"].tolist(), codebert_threshold
        )
        cb_preds = predictions_at_threshold(test_cb, codebert_threshold)
        print(f"CodeBERT threshold (val): {codebert_threshold:.2f} (val F1={codebert_val_f1:.4f})")
        print(
            f"CodeBERT test (n={len(test_df)}) — acc={codebert_test_metrics['accuracy']:.4f} "
            f"f1={codebert_test_metrics['f1']:.4f} "
            f"roc_auc={codebert_test_metrics.get('roc_auc')}"
        )
        _write_method_report(
            metrics_dir,
            "codebert",
            codebert_test_metrics,
            codebert_threshold,
            codebert_val_f1,
            test_df["label"].tolist(),
            cb_preds,
        )
    else:
        print("WARNING: CodeBERT encoder unavailable; skipping CodeBERT similarity metrics.")

    _write_method_report(
        metrics_dir,
        "tfidf",
        tfidf_test_metrics,
        tfidf_threshold,
        tfidf_val_f1,
        test_df["label"].tolist(),
        tfidf_preds,
    )

    threshold_config = {
        "methodology": "threshold_tuned_on_val_evaluated_on_test",
        "grid": {"start": 0.1, "stop": 0.95, "step": 0.05, "objective": "f1"},
        "tfidf": {"validation_threshold": tfidf_threshold, "validation_f1": tfidf_val_f1},
        "codebert": {
            "validation_threshold": codebert_threshold,
            "validation_f1": codebert_val_f1,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_dir / "similarity_threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=2)

    payload = build_similarity_metrics_payload(
        df,
        tfidf_test_metrics,
        tfidf_threshold,
        codebert_test_metrics,
        codebert_threshold,
        tfidf_val_threshold_f1=tfidf_val_f1,
        codebert_val_threshold_f1=codebert_val_f1,
    )
    try:
        from training.dataset_validation import sha256_file

        payload["pairs_checksum_sha256"] = sha256_file(pairs_path)
    except Exception:
        pass

    out = save_similarity_metrics(metrics_dir, payload)
    print(f"[OK] Metrics saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
