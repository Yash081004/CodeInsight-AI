"""
Unified benchmark evaluation (train TF-IDF on train split; report val + optional historical test).

Usage:
    py -3.13 -m training.benchmark_runner --dataset benchmark_v3 --split val
    py -3.13 -m training.benchmark_runner --dataset benchmark_v2 --split test --historical-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID


def collapsed_predictions(y_pred) -> dict:
    c = Counter(ID_TO_LABEL[int(p)] for p in y_pred)
    total = len(y_pred)
    top = c.most_common(1)[0] if c else ("?", 0)
    return {
        "predicted_label_counts": dict(c),
        "dominant_class": top[0],
        "dominant_fraction": round(top[1] / total, 4) if total else 0,
        "collapsed": top[1] / total >= 0.75 if total else False,
        "fault_classes_predicted": sum(1 for lbl in c if lbl != "correct"),
    }


def evaluate_tfidf(df: pd.DataFrame, eval_split: str) -> dict:
    train = df[df["split"] == "train"]
    ev = df[df["split"] == eval_split]
    y_train = train["error_label"].map(LABEL_TO_ID)
    y_ev = ev["error_label"].map(LABEL_TO_ID)
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True, max_features=8000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train["code"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")
    clf.fit(X_train, y_train)
    X_ev = vec.transform(ev["code"])
    pred = clf.predict(X_ev)
    proba = clf.predict_proba(X_ev) if hasattr(clf, "predict_proba") else None
    labels_idx = list(range(len(ERROR_LABELS)))
    report = classification_report(y_ev, pred, labels=labels_idx, target_names=ERROR_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_ev, pred, labels=labels_idx)
    fault_mask = y_ev != LABEL_TO_ID["correct"]
    fp_correct = int(((pred == LABEL_TO_ID["correct"]) & (y_ev != LABEL_TO_ID["correct"])).sum())
    fault_recall = float(f1_score(y_ev[fault_mask], pred[fault_mask], average="macro", zero_division=0)) if fault_mask.any() else None
    fam_metrics = {}
    for fam, grp in ev.groupby("family_id"):
        if len(grp) < 2:
            continue
        yt = grp["error_label"].map(LABEL_TO_ID)
        yp = clf.predict(vec.transform(grp["code"]))
        fam_metrics[fam] = {"n": len(grp), "accuracy": float((yt.values == yp).mean())}
    return {
        "model": "tfidf_logreg",
        "eval_split": eval_split,
        "sample_count": len(ev),
        "accuracy": round(float((pred == y_ev.values).mean()), 4),
        "macro_f1": round(float(f1_score(y_ev, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_ev, pred, average="weighted", zero_division=0)), 4),
        "per_class_metrics": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall": round(report[lbl]["recall"], 4),
                "f1": round(report[lbl]["f1-score"], 4),
                "support": int(report[lbl]["support"]),
            }
            for lbl in ERROR_LABELS
        },
        "confusion_matrix": {"labels": ERROR_LABELS, "matrix": cm.tolist()},
        "collapse": collapsed_predictions(pred),
        "fault_only_macro_f1_proxy": fault_recall,
        "correct_code_false_positive_count": fp_correct,
        "per_family": fam_metrics,
        "probability_note": "Logistic probabilities not calibrated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark_v3", choices=["benchmark_v2", "benchmark_v3"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--historical-test", action="store_true")
    args = parser.parse_args()

    csv = PROJECT_ROOT / "data" / args.dataset / "classification" / "dataset.csv"
    if not csv.exists():
        print(f"Missing {csv}")
        return 1
    df = pd.read_csv(csv)
    eval_split = args.split
    label_note = None
    if eval_split == "test" and args.dataset == "benchmark_v2":
        label_note = "historical_test_previously_inspected"
    if args.historical_test:
        eval_split = "test"
        label_note = label_note or "historical_benchmark_test"

    metrics = evaluate_tfidf(df, eval_split)
    metrics["dataset"] = args.dataset
    metrics["eval_label"] = label_note or "validation_or_holdout"
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = PROJECT_ROOT / "artifacts" / "metrics" / args.dataset / "runner"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tfidf_{eval_split}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[OK] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
