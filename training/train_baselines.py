"""
CodeInsight AI - Baseline Model Training

Trains TF-IDF + Logistic Regression for classification
and TF-IDF vectorizer for similarity baseline.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL


def train_classification_baseline(
    data_path: Path,
    output_dir: Path,
    metrics_dir: Path,
):
    """Train TF-IDF + Logistic Regression classifier."""
    print("\n" + "=" * 60)
    print("Training Classification Baseline")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['error_label'].value_counts().to_dict()}")
    
    # Encode labels
    df["label_id"] = df["error_label"].map(LABEL_TO_ID)
    
    # Split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    if len(train_df) == 0:
        print("ERROR: No training data available.")
        return None
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
    )
    
    X_train = vectorizer.fit_transform(train_df["code"])
    y_train = train_df["label_id"].values
    
    # Handle cases where val or test may be empty
    X_val = vectorizer.transform(val_df["code"]) if len(val_df) > 0 else None
    y_val = val_df["label_id"].values if len(val_df) > 0 else None
    
    X_test = vectorizer.transform(test_df["code"]) if len(test_df) > 0 else None
    y_test = test_df["label_id"].values if len(test_df) > 0 else None
    
    # Train Logistic Regression with class weights
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on test set (or val if no test)
    eval_X = X_test if X_test is not None else X_val
    eval_y = y_test if y_test is not None else y_val
    eval_name = "test" if X_test is not None else "val"
    
    if eval_X is not None:
        y_pred = clf.predict(eval_X)
        
        acc = accuracy_score(eval_y, y_pred)
        macro_p = precision_score(eval_y, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(eval_y, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(eval_y, y_pred, average="macro", zero_division=0)
        
        print(f"\n{eval_name.upper()} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Macro P:   {macro_p:.4f}")
        print(f"  Macro R:   {macro_r:.4f}")
        print(f"  Macro F1:  {macro_f1:.4f}")
        
        # Per-class metrics
        report = classification_report(
            eval_y, y_pred,
            target_names=ERROR_LABELS,
            output_dict=True,
            zero_division=0,
        )
        
        per_class = {}
        for label in ERROR_LABELS:
            if label in report:
                per_class[label] = {
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
        
        # Save metrics
        from training.dataset_validation import sha256_file

        metrics = {
            "model_type": "baseline_logreg",
            "eval_split": eval_name,
            "preprocessing_fit_split": "train",
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_metrics": per_class,
            "sample_count": len(eval_y),
            "evaluated_at": pd.Timestamp.now().isoformat(),
            "dataset_checksum_sha256": sha256_file(data_path) if data_path.exists() else None,
        }
        
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "baseline_classification_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(metrics_dir / "baseline_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        eval_config = {
            "model": "tfidf_logistic_regression",
            "vectorizer_fit": "train_only",
            "classifier_fit": "train_only",
            "evaluation_split": eval_name,
            "random_state": 42,
            "class_weight": "balanced",
        }
        with open(metrics_dir / "baseline_eval_config.json", "w") as f:
            json.dump(eval_config, f, indent=2)

        # Save confusion matrix
        cm = confusion_matrix(eval_y, y_pred)
        cm_data = {
            "labels": ERROR_LABELS,
            "matrix": cm.tolist(),
            "model_type": "baseline",
        }
        with open(metrics_dir / "baseline_confusion_matrix.json", "w") as f:
            json.dump(cm_data, f, indent=2)

        try:
            from training.evaluation_artifacts import save_confusion_matrix_figure

            save_confusion_matrix_figure(
                cm.tolist(),
                ERROR_LABELS,
                f"Baseline classification ({eval_name}, n={len(eval_y)})",
                metrics_dir.parent / "figures" / "baseline_classification_confusion.png",
            )
        except Exception as fig_err:
            print(f"  (figure save skipped: {fig_err})")
        
        print(f"\n  Classification Report:")
        print(classification_report(eval_y, y_pred, target_names=ERROR_LABELS, zero_division=0))
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "classification_logreg.joblib")
    joblib.dump(vectorizer, output_dir / "classification_tfidf_vectorizer.joblib")
    
    # Save label mapping
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump({"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL}, f, indent=2)
    
    print(f"\n[OK] Baseline classifier saved to {output_dir}")
    return metrics if eval_X is not None else None


def train_similarity_baseline(
    data_path: Path,
    output_dir: Path,
    metrics_dir: Path,
):
    """Train TF-IDF vectorizer for similarity baseline."""
    print("\n" + "=" * 60)
    print("Training Similarity Baseline")
    print("=" * 60)
    
    # Load classification dataset for vocabulary
    clf_path = data_path.parent / "classification" / "dataset.csv"
    if clf_path.exists():
        df = pd.read_csv(clf_path)
        corpus = df["code"].tolist()
    else:
        print("WARNING: No classification dataset found for TF-IDF fitting.")
        corpus = []
    
    # Also load similarity pairs if available
    pairs_path = data_path / "pairs.csv"
    if pairs_path.exists():
        pairs_df = pd.read_csv(pairs_path)
        corpus.extend(pairs_df["code_a"].tolist())
        corpus.extend(pairs_df["code_b"].tolist())
        print(f"Similarity pairs: {len(pairs_df)}")
    
    if not corpus:
        print("ERROR: No data for TF-IDF training.")
        return None
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
    )
    vectorizer.fit(corpus)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "similarity_tfidf_vectorizer.joblib")
    
    print(f"[OK] TF-IDF vectorizer saved ({len(corpus)} documents, {len(vectorizer.vocabulary_)} features)")
    
    # Evaluate on similarity pairs if available (val threshold, test metrics)
    if pairs_path.exists():
        from training.similarity_evaluation import (
            build_similarity_metrics_payload,
            find_best_threshold,
            load_similarity_pairs,
            metrics_at_threshold,
            save_similarity_metrics,
            split_frame,
            tfidf_pair_similarities,
        )

        pairs_df = load_similarity_pairs(pairs_path)
        val_df, test_df = split_frame(pairs_df)

        if len(val_df) == 0 or len(test_df) == 0:
            print("WARNING: Skipping similarity metrics — need val and test splits.")
        else:
            val_sims = tfidf_pair_similarities(
                vectorizer,
                val_df["code_a"].tolist(),
                val_df["code_b"].tolist(),
            )
            test_sims = tfidf_pair_similarities(
                vectorizer,
                test_df["code_a"].tolist(),
                test_df["code_b"].tolist(),
            )
            best_threshold, _ = find_best_threshold(val_sims, val_df["label"].tolist())
            test_metrics = metrics_at_threshold(
                test_sims, test_df["label"].tolist(), best_threshold
            )

            sim_metrics = build_similarity_metrics_payload(
                pairs_df,
                test_metrics,
                best_threshold,
            )

            save_similarity_metrics(metrics_dir, sim_metrics)

            acc = test_metrics["accuracy"]
            p = test_metrics["precision"]
            r = test_metrics["recall"]
            f1 = test_metrics["f1"]

            print(f"\n  Similarity Metrics (threshold={best_threshold:.2f} from val, test n={len(test_df)}):")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {p:.4f}")
            print(f"  Recall:    {r:.4f}")
            print(f"  F1:        {f1:.4f}")
            print("  Run `py -3.13 -m training.evaluate_similarity` for CodeBERT pair metrics.")
    
    return None


if __name__ == "__main__":
    project_root = PROJECT_ROOT
    data_dir = project_root / "data"
    models_dir = project_root / "models" / "baselines"
    metrics_dir = project_root / "artifacts" / "metrics"
    
    # Train classification baseline
    clf_data = data_dir / "classification" / "dataset.csv"
    if clf_data.exists():
        train_classification_baseline(clf_data, models_dir, metrics_dir)
    else:
        print(f"Classification dataset not found at {clf_data}")
        print("Run: python -m training.generate_dataset")
    
    # Train similarity baseline
    train_similarity_baseline(data_dir / "similarity", models_dir, metrics_dir)
    
    print("\n" + "=" * 60)
    print("Baseline training complete!")
