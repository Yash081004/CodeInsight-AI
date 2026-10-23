"""
Phase 10 controlled experiments E0 (TF-IDF) and E1 (CodeBERT) on benchmark v3.

Usage:
  py -3.13 -m training.phase10_experiments --e0
  py -3.13 -m training.phase10_experiments --e1
  py -3.13 -m training.phase10_experiments --e0 --e1 --acceptance-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID
from training.dataset_validation import sha256_file
from training.phase10_v3_exclusions import (
    apply_train_val_template_exclusion,
    exclusion_manifest,
    load_review_queue_template_ids,
    verify_no_family_leakage,
    write_filtered_view_csv,
)

V3_CSV = PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv"
V3_MANIFEST = PROJECT_ROOT / "data/benchmark_v3/manifest.json"
REVIEW_QUEUE = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
PHASE10_METRICS = PROJECT_ROOT / "artifacts/metrics/phase10"
E0_METRICS = PHASE10_METRICS / "E0_tfidf_v3"
E1_METRICS = PHASE10_METRICS / "E1_codebert_v3"
E0_MODELS = PROJECT_ROOT / "models/experiments/phase10_e0_tfidf_v3"
E1_MODELS = PROJECT_ROOT / "models/experiments/phase10_e1_codebert_v3"

GATE = {
    "macro_f1_min": 0.35,
    "fault_recall_min": 0.25,
    "correct_recall_min": 0.50,
    "max_single_class_pred_fraction": 0.70,
}


def _metrics_bundle(y_true, y_pred, split_name: str) -> Dict[str, Any]:
    labels_idx = list(range(len(ERROR_LABELS)))
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=ERROR_LABELS, output_dict=True, zero_division=0
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
    pred_labels = [ID_TO_LABEL[int(p)] for p in y_pred]
    dist = Counter(pred_labels)
    n = max(len(y_pred), 1)
    max_frac = max(dist.values()) / n if dist else 0.0
    return {
        "eval_split": split_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class,
        "predicted_label_distribution": dict(dist),
        "max_predicted_class_fraction": round(max_frac, 4),
        "collapse_toward_correct": dist.get("correct", 0) / n >= 0.75,
        "sample_count": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_idx).tolist(),
    }


def evaluate_acceptance_gate(val_metrics: Dict[str, Any], queue_excluded_from_train: bool) -> Dict[str, Any]:
    per = val_metrics.get("per_class_metrics") or {}
    checks = {
        "macro_f1": val_metrics.get("macro_f1", 0) >= GATE["macro_f1_min"],
        "correct_recall": per.get("correct", {}).get("recall", 0) >= GATE["correct_recall_min"],
        "max_class_fraction": val_metrics.get("max_predicted_class_fraction", 1)
        <= GATE["max_single_class_pred_fraction"],
        "review_queue_excluded_from_train_val": queue_excluded_from_train,
    }
    for fault in ERROR_LABELS:
        if fault == "correct":
            continue
        checks[f"recall_{fault}"] = per.get(fault, {}).get("recall", 0) >= GATE["fault_recall_min"]
    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "thresholds": GATE}


def run_e0() -> Dict[str, Any]:
    E0_METRICS.mkdir(parents=True, exist_ok=True)
    E0_MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(V3_CSV)
    checksum = sha256_file(V3_CSV)
    df["label_id"] = df["error_label"].map(LABEL_TO_ID)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_df["code"])
    y_train = train_df["label_id"].values
    X_val = vectorizer.transform(val_df["code"])
    y_val = val_df["label_id"].values
    X_test = vectorizer.transform(test_df["code"])
    y_test = test_df["label_id"].values

    c_grid = [0.01, 0.1, 1.0, 10.0]
    best_c, best_val_f1 = 1.0, -1.0
    for c in c_grid:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=c, solver="lbfgs")
        clf.fit(X_train, y_train)
        pred_v = clf.predict(X_val)
        vf1 = f1_score(y_val, pred_v, average="macro", zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_c = c

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=best_c, solver="lbfgs")
    clf.fit(X_train, y_train)
    val_m = _metrics_bundle(y_val, clf.predict(X_val), "val")
    test_m = _metrics_bundle(y_test, clf.predict(X_test), "test")

    joblib.dump(clf, E0_MODELS / "classification_logreg.joblib")
    joblib.dump(vectorizer, E0_MODELS / "classification_tfidf_vectorizer.joblib")

    runtime_s = round(time.time() - t0, 2)
    payload = {
        "experiment_id": "E0_tfidf_v3",
        "model_type": "tfidf_logreg_baseline",
        "dataset_path": str(V3_CSV.relative_to(PROJECT_ROOT)),
        "dataset_checksum_sha256": checksum,
        "manifest_checksum_sha256": json.loads(V3_MANIFEST.read_text()).get("checksum_sha256"),
        "hyperparameter_selection": {"split": "val", "C_grid": c_grid, "selected_C": best_c, "selected_val_macro_f1": round(best_val_f1, 4)},
        "random_state": 42,
        "validation": val_m,
        "test": test_m,
        "runtime_seconds": runtime_s,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_gate": evaluate_acceptance_gate(val_m, queue_excluded_from_train=False),
    }
    (E0_METRICS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (E0_METRICS / "validation_confusion_matrix.json").write_text(
        json.dumps({"labels": ERROR_LABELS, "matrix": val_m["confusion_matrix"]}, indent=2),
        encoding="utf-8",
    )
    (E0_METRICS / "test_confusion_matrix.json").write_text(
        json.dumps({"labels": ERROR_LABELS, "matrix": test_m["confusion_matrix"]}, indent=2),
        encoding="utf-8",
    )
    print(f"[E0] val macro F1={val_m['macro_f1']} test macro F1={test_m['macro_f1']} ({runtime_s}s)")
    return payload


def run_e1() -> Dict[str, Any]:
    E1_METRICS.mkdir(parents=True, exist_ok=True)
    E1_MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    full_checksum = sha256_file(V3_CSV)
    df = pd.read_csv(V3_CSV)
    template_ids = load_review_queue_template_ids(REVIEW_QUEUE)
    filtered, excluded = apply_train_val_template_exclusion(df, template_ids)
    ok, leak_errs = verify_no_family_leakage(filtered)
    if not ok:
        raise RuntimeError(f"Family leakage after filter: {leak_errs[:5]}")

    view_path = PHASE10_METRICS / "E1_filtered_dataset_view.csv"
    view_checksum = write_filtered_view_csv(filtered, view_path)
    manifest = exclusion_manifest(full_checksum, REVIEW_QUEUE, template_ids, excluded, filtered)
    manifest["filtered_view_path"] = str(view_path.relative_to(PROJECT_ROOT))
    manifest["filtered_view_checksum_sha256"] = view_checksum
    (E1_METRICS / "exclusion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from training.train_codebert_classifier import train

    class Args:
        data_path = str(view_path)
        output_dir = str(E1_MODELS)
        metrics_dir = str(E1_METRICS)
        model_name = "microsoft/codebert-base"
        epochs = 3
        batch_size = 4
        eval_batch_size = 8
        lr = 2e-5
        max_length = 256
        seed = 42
        skip_preflight = True  # filtered view checksum != v3 manifest; leakage checked in phase10
        no_class_weights = False
        balanced_sampler = False
        skip_test_eval = False
        experiment_id = "E1_codebert_v3"
        patience = 0
        exclude_templates_from = None

    train(Args())

    runtime_s = round(time.time() - t0, 2)
    val_path = E1_METRICS / "validation_metrics.json"
    test_path = E1_METRICS / "codebert_classification_metrics.json"
    val_m = json.loads(val_path.read_text()) if val_path.is_file() else {}
    test_m = json.loads(test_path.read_text()) if test_path.is_file() else {}

    # Normalize for acceptance
    if val_m and "predicted_label_counts" in val_m:
        total = sum(val_m["predicted_label_counts"].values()) or 1
        val_m["max_predicted_class_fraction"] = max(val_m["predicted_label_counts"].values()) / total
        val_m["macro_f1"] = val_m.get("macro_f1", 0)

    summary = {
        "experiment_id": "E1_codebert_v3",
        "exclusion_manifest": manifest,
        "runtime_seconds": runtime_s,
        "validation_metrics_file": str(val_path.relative_to(PROJECT_ROOT)),
        "test_metrics_file": str(test_path.relative_to(PROJECT_ROOT)) if test_path.is_file() else None,
        "acceptance_gate": evaluate_acceptance_gate(
            {
                "macro_f1": val_m.get("macro_f1", 0),
                "per_class_metrics": val_m.get("per_class_metrics", {}),
                "max_predicted_class_fraction": val_m.get("max_predicted_class_fraction", 1),
            },
            queue_excluded_from_train=True,
        ),
        "test_set_caveat": "benchmark_v3 test used once for final reporting; not used for checkpoint selection.",
    }
    (E1_METRICS / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[E1] val macro F1={val_m.get('macro_f1')} ({runtime_s}s total)")
    return summary


def write_acceptance_report(e0: Optional[Dict], e1: Optional[Dict]) -> Path:
    docs = PROJECT_ROOT / "docs/phase10"
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 10 — E0/E1 validation comparison & acceptance gate",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## E0 TF-IDF (benchmark v3, full train/val/test)",
        "",
    ]
    if e0:
        lines.append(f"- Val macro F1: **{e0['validation']['macro_f1']}**")
        lines.append(f"- Test macro F1: **{e0['test']['macro_f1']}** (single post-selection eval)")
        lines.append(f"- Gate passed: **{e0['acceptance_gate']['passed']}**")
        lines.append(f"- Selected C: {e0['hyperparameter_selection']['selected_C']}")
    if e1:
        lines.extend(["", "## E1 CodeBERT (review-queue templates excluded from train/val)", ""])
        vm = json.loads((E1_METRICS / "validation_metrics.json").read_text())
        lines.append(f"- Val macro F1: **{vm.get('macro_f1')}**")
        if (E1_METRICS / "codebert_classification_metrics.json").is_file():
            tm = json.loads((E1_METRICS / "codebert_classification_metrics.json").read_text())
            lines.append(f"- Test macro F1: **{tm.get('macro_f1')}** (checkpoint from val only)")
        lines.append(f"- Gate passed: **{e1['acceptance_gate']['passed']}**")
        lines.append(f"- Excluded rows: {e1['exclusion_manifest']['excluded_row_count']}")

    lines.extend(
        [
            "",
            "## Acceptance thresholds (Phase 9)",
            "",
            f"- Macro F1 ≥ {GATE['macro_f1_min']}",
            f"- Each fault recall ≥ {GATE['fault_recall_min']}",
            f"- Correct recall ≥ {GATE['correct_recall_min']}",
            f"- Max single-class prediction share ≤ {GATE['max_single_class_pred_fraction']}",
            "- Review-queue templates excluded from E1 train/val",
            "",
            "## Promotion",
            "",
            "Neither candidate replaces `models/codebert_classifier/` unless all gate checks pass on **validation**.",
        ]
    )
    path = docs / "ACCEPTANCE_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (PHASE10_METRICS / "acceptance_report.json").write_text(
        json.dumps({"e0": e0, "e1": e1, "gate": GATE}, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", action="store_true")
    parser.add_argument("--e1", action="store_true")
    parser.add_argument("--acceptance-report", action="store_true")
    args = parser.parse_args()
    if not (args.e0 or args.e1):
        args.e0 = args.e1 = True

    e0 = run_e0() if args.e0 else None
    e1 = run_e1() if args.e1 else None
    if args.acceptance_report or args.e0 or args.e1:
        p = write_acceptance_report(e0, e1)
        print(f"Wrote {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
