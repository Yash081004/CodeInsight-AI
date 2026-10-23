"""
Shared similarity pair evaluation utilities.

Threshold selection uses validation split only; reported metrics use test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def load_similarity_pairs(pairs_path: Path) -> pd.DataFrame:
    """Load pairs CSV, excluding cross-split leakage rows."""
    df = pd.read_csv(pairs_path)
    if "split" in df.columns:
        df = df[df["split"] != "cross_split"].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def find_best_threshold(sims: List[float], labels: List[int]) -> Tuple[float, float]:
    """Grid-search threshold maximizing F1 on the provided split (validation)."""
    best_f1 = -1.0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = [1 if s >= t else 0 for s in sims]
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_threshold = float(t)
    return best_threshold, best_f1


def metrics_at_threshold(sims: List[float], labels: List[int], threshold: float) -> Dict[str, float]:
    preds = [1 if s >= threshold else 0 for s in sims]
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
        "sample_count": len(labels),
    }
    roc = safe_roc_auc(labels, sims)
    if roc is not None:
        out["roc_auc"] = roc
    return out


def predictions_at_threshold(sims: List[float], threshold: float) -> List[int]:
    return [1 if s >= threshold else 0 for s in sims]


def safe_roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """ROC-AUC on similarity scores (not calibrated probabilities)."""
    if len(set(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def tfidf_pair_similarities(vectorizer, codes_a: List[str], codes_b: List[str]) -> List[float]:
    sims = []
    for ca, cb in zip(codes_a, codes_b):
        vecs = vectorizer.transform([ca, cb])
        sims.append(float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0]))
    return sims


def split_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return val_df, test_df


def build_similarity_metrics_payload(
    df: pd.DataFrame,
    tfidf_test: Dict[str, float],
    tfidf_threshold: float,
    codebert_test: Optional[Dict[str, float]] = None,
    codebert_threshold: Optional[float] = None,
    tfidf_val_threshold_f1: Optional[float] = None,
    codebert_val_threshold_f1: Optional[float] = None,
) -> Dict:
    labels = df["label"].tolist()
    payload = {
        "methodology": "threshold_tuned_on_val_evaluated_on_test",
        "threshold_selection": "grid_search_0.05_step_maximize_f1_on_val",
        "scores_are_calibrated_probabilities": False,
        "tfidf_threshold": round(tfidf_threshold, 2),
        "tfidf_val_threshold_f1": round(tfidf_val_threshold_f1, 4) if tfidf_val_threshold_f1 is not None else None,
        "tfidf_test_accuracy": round(tfidf_test["accuracy"], 4),
        "tfidf_test_precision": round(tfidf_test["precision"], 4),
        "tfidf_test_recall": round(tfidf_test["recall"], 4),
        "tfidf_test_f1": round(tfidf_test["f1"], 4),
        "tfidf_test_roc_auc": round(tfidf_test["roc_auc"], 4) if tfidf_test.get("roc_auc") is not None else None,
        "tfidf_test_sample_count": int(tfidf_test.get("sample_count", 0)),
        "codebert_threshold": round(codebert_threshold, 2) if codebert_threshold is not None else None,
        "codebert_val_threshold_f1": round(codebert_val_threshold_f1, 4) if codebert_val_threshold_f1 is not None else None,
        "codebert_test_accuracy": round(codebert_test["accuracy"], 4) if codebert_test else None,
        "codebert_test_precision": round(codebert_test["precision"], 4) if codebert_test else None,
        "codebert_test_recall": round(codebert_test["recall"], 4) if codebert_test else None,
        "codebert_test_f1": round(codebert_test["f1"], 4) if codebert_test else None,
        "codebert_test_roc_auc": round(codebert_test["roc_auc"], 4) if codebert_test and codebert_test.get("roc_auc") is not None else None,
        "codebert_test_sample_count": int(codebert_test.get("sample_count", 0)) if codebert_test else 0,
        # Backward-compatible keys for frontend
        "tfidf_accuracy": round(tfidf_test["accuracy"], 4),
        "tfidf_precision": round(tfidf_test["precision"], 4),
        "tfidf_recall": round(tfidf_test["recall"], 4),
        "tfidf_f1": round(tfidf_test["f1"], 4),
        "codebert_accuracy": round(codebert_test["accuracy"], 4) if codebert_test else None,
        "codebert_precision": round(codebert_test["precision"], 4) if codebert_test else None,
        "codebert_recall": round(codebert_test["recall"], 4) if codebert_test else None,
        "codebert_f1": round(codebert_test["f1"], 4) if codebert_test else None,
        "val_pairs": int((df["split"] == "val").sum()),
        "test_pairs": int((df["split"] == "test").sum()),
        "cross_split_pairs_excluded": int((df["split"] == "cross_split").sum()) if "split" in df.columns else 0,
        "total_pairs": len(df),
        "similar_pairs": int(sum(labels)),
        "dissimilar_pairs": int(len(labels) - sum(labels)),
    }
    return payload


def save_similarity_metrics(metrics_dir: Path, payload: Dict) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "similarity_metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out
