"""
Train/evaluate baselines on benchmark_v2 without touching production models.

Usage:
    py -3.13 -m training.train_eval_benchmark_v2
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL
from training.dataset_pipeline import DATASET_VERSION
from training.dataset_pipeline.similarity_v2 import code_hash
from training.similarity_evaluation import (
    find_best_threshold,
    load_similarity_pairs,
    metrics_at_threshold,
    split_frame,
    tfidf_pair_similarities,
)


def train_classification_v2(data_path: Path, out_dir: Path) -> dict:
    df = pd.read_csv(data_path)
    df["label_id"] = df["error_label"].map(LABEL_TO_ID)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True, max_features=8000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_df["code"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")
    clf.fit(X_train, train_df["label_id"])

    X_test = vec.transform(test_df["code"])
    y_test = test_df["label_id"].values
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=ERROR_LABELS, output_dict=True, zero_division=0)

    per_class = {}
    for label in ERROR_LABELS:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
            }

    manifest_path = data_path.parent.parent / "manifest.json"
    checksum = None
    if manifest_path.exists():
        with open(manifest_path) as mf:
            checksum = json.load(mf).get("checksum_sha256")

    metrics = {
        "model_type": "tfidf_logreg_baseline",
        "dataset_version": DATASET_VERSION,
        "dataset_checksum_sha256": checksum,
        "eval_split": "test",
        "train_n": len(train_df),
        "val_n": len(val_df),
        "test_n": len(test_df),
        "sample_count": len(test_df),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "per_class_metrics": per_class,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "classification_logreg.joblib")
    joblib.dump(vec, out_dir / "classification_tfidf_vectorizer.joblib")

    cm = confusion_matrix(y_test, y_pred).tolist()
    with open(out_dir / "confusion_matrix.json", "w") as f:
        json.dump({"labels": ERROR_LABELS, "matrix": cm}, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    preds = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        probs = None
        if proba is not None:
            probs = {ID_TO_LABEL[j]: round(float(proba[i, j]), 4) for j in range(len(ERROR_LABELS))}
        preds.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row.get("family_id"),
                "true_label": row["error_label"],
                "predicted_id": int(y_pred[i]),
                "predicted_label": ID_TO_LABEL[int(y_pred[i])],
                "probabilities": probs,
            }
        )
    with open(out_dir / "test_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)

    with open(out_dir / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def eval_similarity_v2(pairs_path: Path, vec_path: Path, out_dir: Path) -> dict:
    df = load_similarity_pairs(pairs_path)
    val_df, test_df = split_frame(df)
    vec = joblib.load(vec_path)

    val_sims = tfidf_pair_similarities(vec, val_df["code_a"].tolist(), val_df["code_b"].tolist())
    test_sims = tfidf_pair_similarities(vec, test_df["code_a"].tolist(), test_df["code_b"].tolist())
    thr, _ = find_best_threshold(val_sims, val_df["label"].tolist())
    test_m = metrics_at_threshold(test_sims, test_df["label"].tolist(), thr)

    payload = {
        "dataset_version": DATASET_VERSION,
        "method": "tfidf_cosine",
        "val_pairs": len(val_df),
        "test_pairs": len(test_df),
        "threshold_val": thr,
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    with open(out_dir / "similarity_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> int:
    data_root = PROJECT_ROOT / "data" / DATASET_VERSION
    clf_csv = data_root / "classification" / "dataset.csv"
    pairs_csv = data_root / "similarity" / "pairs.csv"
    if not clf_csv.exists():
        print("Run: py -3.13 -m training.dataset_pipeline.generate_benchmark")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / run_id
    models_dir = PROJECT_ROOT / "models" / "benchmark_v2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / "manifest.json") as f:
        manifest = json.load(f)
    config = {
        "dataset_version": DATASET_VERSION,
        "checksum": manifest.get("checksum_sha256"),
        "seed": manifest.get("seed"),
        "run_id": run_id,
        "production_checkpoint_touched": False,
    }
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Training TF-IDF + LogReg on benchmark_v2 train split only...")
    clf_metrics = train_classification_v2(clf_csv, out_dir)
    for name in ("classification_logreg.joblib", "classification_tfidf_vectorizer.joblib"):
        joblib.dump(joblib.load(out_dir / name), models_dir / name)
    print(clf_metrics)

    from training.hybrid_classifier import evaluate_hybrid_on_test

    hybrid_metrics = evaluate_hybrid_on_test(clf_csv, out_dir / "classification_logreg.joblib", out_dir / "classification_tfidf_vectorizer.joblib")
    with open(out_dir / "hybrid_classification_metrics.json", "w") as f:
        json.dump(hybrid_metrics, f, indent=2)
    print({"hybrid": hybrid_metrics})

    # Fit similarity vectorizer on train classification codes only
    df = pd.read_csv(clf_csv)
    train_codes = df[df["split"] == "train"]["code"].tolist()
    pairs_df = pd.read_csv(pairs_csv)
    corpus = train_codes + pairs_df[pairs_df["split"] != "cross_split"]["code_a"].tolist()
    corpus += pairs_df[pairs_df["split"] != "cross_split"]["code_b"].tolist()
    sim_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True, max_features=8000, ngram_range=(1, 2))
    sim_vec.fit(corpus)
    joblib.dump(sim_vec, models_dir / "similarity_tfidf_vectorizer.joblib")

    if pairs_csv.exists():
        sim_m = eval_similarity_v2(pairs_csv, models_dir / "similarity_tfidf_vectorizer.joblib", out_dir)
        print(sim_m)
        try:
            from backend.app.config import get_settings
            from backend.app.models.codebert_encoder import CodeBERTEncoder
            from training.evaluate_similarity import codebert_pair_similarities

            settings = get_settings()
            enc = CodeBERTEncoder(model_name=settings.codebert_model_name, device=settings.device)
            if enc.load():
                df = load_similarity_pairs(pairs_csv)
                val_df, test_df = split_frame(df)
                val_cb = codebert_pair_similarities(enc, val_df["code_a"].tolist(), val_df["code_b"].tolist())
                test_cb = codebert_pair_similarities(enc, test_df["code_a"].tolist(), test_df["code_b"].tolist())
                thr, _ = find_best_threshold(val_cb, val_df["label"].tolist())
                cb_m = metrics_at_threshold(test_cb, test_df["label"].tolist(), thr)
                cb_payload = {
                    "dataset_version": DATASET_VERSION,
                    "method": "codebert_cosine",
                    "val_pairs": len(val_df),
                    "test_pairs": len(test_df),
                    "threshold_val": thr,
                    **{f"test_{k}": v for k, v in cb_m.items()},
                }
                with open(out_dir / "similarity_codebert_metrics.json", "w") as f:
                    json.dump(cb_payload, f, indent=2)
                print(cb_payload)
        except Exception as exc:
            print(f"[WARN] CodeBERT similarity eval skipped: {exc}")

    print(f"[OK] artifacts: {out_dir}")
    print(f"[OK] models: {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
