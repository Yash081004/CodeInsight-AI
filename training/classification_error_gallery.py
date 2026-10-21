"""
Build classification error gallery artifacts for benchmark_v2 runs.

Usage:
    py -3.13 -m training.classification_error_gallery --run-id codebert_20261019_120000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID
from training.dataset_pipeline import DATASET_VERSION

CONFUSION_PAIRS = [
    ("off_by_one_loop", "wrong_condition"),
    ("wrong_operator", "wrong_initialization"),
    ("wrong_condition", "correct"),
    ("wrong_initialization", "correct"),
]


def _error_category(true_label: str, pred_label: str) -> str:
    if true_label == pred_label:
        return "correct_prediction"
    if true_label == "correct":
        return "false_positive_fault"
    if pred_label == "correct":
        return "missed_fault"
    for a, b in CONFUSION_PAIRS:
        if (true_label, pred_label) in ((a, b), (b, a)):
            return f"confusion_{a}_vs_{b}"
    return "other_misclassification"


def _serialize_code(code: str, max_len: int = 4000) -> str:
    if len(code) <= max_len:
        return code
    return code[: max_len - 20] + "\n# ... truncated ..."


def _record_from_row(
    row: pd.Series,
    pred_label: str,
    probabilities: Optional[Dict[str, float]],
    model_id: str,
    experiment_id: str,
) -> dict:
    true_label = row["error_label"]
    correct = true_label == pred_label
    prob_note = (
        "Softmax scores from model head — not calibrated probabilities."
        if probabilities
        else None
    )
    return {
        "sample_id": row["sample_id"],
        "family_id": row.get("family_id", ""),
        "source_template_id": row.get("source_template_id", ""),
        "true_label": true_label,
        "predicted_label": pred_label,
        "correct": correct,
        "probabilities": probabilities,
        "probability_note": prob_note,
        "model_id": model_id,
        "experiment_id": experiment_id,
        "code": _serialize_code(str(row["code"])),
        "error_category": _error_category(true_label, pred_label),
    }


def baseline_predictions(
    dataset_csv: Path,
    clf_path: Path,
    vec_path: Path,
    experiment_id: str,
) -> List[dict]:
    df = pd.read_csv(dataset_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)
    X = vec.transform(test_df["code"])
    pred_ids = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
    else:
        proba = None
    out = []
    for i, row in test_df.iterrows():
        probs = None
        if proba is not None:
            probs = {
                ID_TO_LABEL[j]: round(float(proba[i, j]), 4)
                for j in range(len(ERROR_LABELS))
            }
        out.append(
            _record_from_row(
                row,
                ID_TO_LABEL[int(pred_ids[i])],
                probs,
                "tfidf_logreg_baseline",
                experiment_id,
            )
        )
    return out


def load_predictions_json(path: Path, model_id: str, experiment_id: str, dataset_csv: Path) -> List[dict]:
    df = pd.read_csv(dataset_csv)
    test_df = df[df["split"] == "test"].set_index("sample_id")
    with open(path) as f:
        raw = json.load(f)
    out = []
    for item in raw:
        sid = item["sample_id"]
        if sid not in test_df.index:
            continue
        row = test_df.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["sample_id"] = sid
        probs = item.get("probabilities")
        out.append(
            _record_from_row(
                row,
                item.get("predicted_label") or ID_TO_LABEL[int(item["predicted_id"])],
                probs,
                model_id,
                experiment_id,
            )
        )
    return out


def build_per_class_analysis(records: List[dict]) -> dict:
    by_class: Dict[str, dict] = {}
    for label in ERROR_LABELS:
        subset = [r for r in records if r["true_label"] == label]
        if not subset:
            by_class[label] = {"support": 0, "errors": 0, "top_confusions": []}
            continue
        errors = [r for r in subset if not r["correct"]]
        conf = Counter((r["true_label"], r["predicted_label"]) for r in errors)
        by_class[label] = {
            "support": len(subset),
            "errors": len(errors),
            "error_rate": round(len(errors) / len(subset), 4),
            "top_confusions": [
                {"predicted_as": p, "count": c}
                for (_, p), c in conf.most_common(5)
            ],
        }
    return by_class


def build_gallery_entries(
    baseline: List[dict],
    codebert: List[dict],
) -> dict:
    by_id_b = {r["sample_id"]: r for r in baseline}
    by_id_c = {r["sample_id"]: r for r in codebert}
    ids = sorted(set(by_id_b) & set(by_id_c))
    entries = []
    for sid in ids:
        b, c = by_id_b[sid], by_id_c[sid]
        entries.append(
            {
                "sample_id": sid,
                "family_id": b["family_id"],
                "true_label": b["true_label"],
                "baseline": {
                    "predicted_label": b["predicted_label"],
                    "correct": b["correct"],
                    "probabilities": b.get("probabilities"),
                },
                "codebert": {
                    "predicted_label": c["predicted_label"],
                    "correct": c["correct"],
                    "probabilities": c.get("probabilities"),
                },
                "both_correct": b["correct"] and c["correct"],
                "baseline_only_wrong": b["correct"] and not c["correct"],
                "codebert_only_wrong": c["correct"] and not b["correct"],
                "both_wrong": not b["correct"] and not c["correct"],
                "code": b["code"],
                "error_category_baseline": b["error_category"],
                "error_category_codebert": c["error_category"],
            }
        )

    def pick(filter_fn, limit=8):
        return [e for e in entries if filter_fn(e)][:limit]

    return {
        "matched_test_ids": ids,
        "total": len(entries),
        "representative": {
            "false_positives_on_correct_code": pick(
                lambda e: e["true_label"] == "correct"
                and (not e["baseline"]["correct"] or not e["codebert"]["correct"])
            ),
            "false_negatives_missed_faults": pick(
                lambda e: e["true_label"] != "correct"
                and (e["baseline"]["predicted_label"] == "correct" or e["codebert"]["predicted_label"] == "correct")
            ),
            "category_confusions": pick(
                lambda e: e["true_label"] != e["baseline"]["predicted_label"]
                and e["true_label"] in ("off_by_one_loop", "wrong_condition", "wrong_operator")
            ),
            "both_models_wrong": pick(lambda e: e["both_wrong"]),
        },
        "by_family": _group_by_family(entries),
        "entries": entries,
    }


def _group_by_family(entries: List[dict]) -> dict:
    fam: Dict[str, list] = defaultdict(list)
    for e in entries:
        fam[e["family_id"]].append(
            {
                "sample_id": e["sample_id"],
                "true_label": e["true_label"],
                "baseline_pred": e["baseline"]["predicted_label"],
                "codebert_pred": e["codebert"]["predicted_label"],
            }
        )
    return dict(fam)


def failure_analysis(records: List[dict], model_id: str) -> dict:
    errors = [r for r in records if not r["correct"]]
    by_true = Counter(r["true_label"] for r in errors)
    by_pred = Counter(r["predicted_label"] for r in errors)
    by_cat = Counter(r["error_category"] for r in errors)
    by_family = Counter(r["family_id"] for r in errors)

    observed = {
        "model_id": model_id,
        "test_errors": len(errors),
        "errors_by_true_label": dict(by_true),
        "errors_by_predicted_label": dict(by_pred),
        "errors_by_category": dict(by_cat),
        "families_with_most_errors": by_family.most_common(10),
    }
    hypotheses = []
    if by_true.get("off_by_one_loop", 0) >= 2:
        hypotheses.append(
            {
                "pattern": "off_by_one_loop misclassification",
                "evidence": f"{by_true['off_by_one_loop']} errors with true label off_by_one_loop",
                "hypothesis": "Loop-boundary mutations may be lexically similar to correct code",
                "smallest_test": "Add explicit range() token features or 1-epoch LR sweep on val only",
            }
        )
    if by_pred.get("correct", 0) > len(errors) // 3:
        hypotheses.append(
            {
                "pattern": "predicted_as_correct",
                "evidence": f"{by_pred['correct']} fault samples predicted as correct",
                "hypothesis": "Class imbalance toward correct may bias the head",
                "smallest_test": "Compare val macro-F1 with stronger class weights (val-only)",
            }
        )
    return {"observed_facts": observed, "hypotheses_not_proven": hypotheses}


def build_model_comparison(
    baseline_metrics: dict,
    codebert_metrics: dict,
    manifest: dict,
    baseline_run: str,
    codebert_run: str,
    test_ids: List[str],
) -> dict:
    test_df_families = manifest.get("families_per_split", {}).get("test", [])
    return {
        "dataset_version": DATASET_VERSION,
        "dataset_checksum_sha256": manifest.get("checksum_sha256"),
        "split_manifest": manifest.get("families_per_split"),
        "baseline_run_id": baseline_run,
        "codebert_run_id": codebert_run,
        "test_sample_count": codebert_metrics.get("sample_count") or codebert_metrics.get("test_n"),
        "test_family_count": len(test_df_families),
        "test_family_ids": test_df_families,
        "matched_test_sample_ids": test_ids,
        "baseline": {
            "macro_f1": baseline_metrics.get("macro_f1"),
            "accuracy": baseline_metrics.get("accuracy"),
            "weighted_f1": baseline_metrics.get("weighted_f1"),
        },
        "codebert": {
            "macro_f1": codebert_metrics.get("macro_f1"),
            "accuracy": codebert_metrics.get("accuracy"),
            "weighted_f1": codebert_metrics.get("weighted_f1"),
            "best_epoch": codebert_metrics.get("best_epoch"),
            "best_val_macro_f1": codebert_metrics.get("best_val_macro_f1"),
        },
        "codebert_beats_baseline_macro_f1": (
            (codebert_metrics.get("macro_f1") or 0) > (baseline_metrics.get("macro_f1") or 0)
        ),
        "note": "Comparison uses identical test sample IDs from benchmark_v2 test split only.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebert-run", required=True, help="e.g. codebert_20261019_120000")
    parser.add_argument(
        "--baseline-run",
        default="20261019_062841",
        help="TF-IDF baseline run under artifacts/metrics/benchmark_v2/",
    )
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data" / DATASET_VERSION
    dataset_csv = data_root / "classification" / "dataset.csv"
    manifest_path = data_root / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    metrics_root = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2"
    cb_dir = metrics_root / args.codebert_run
    bl_dir = metrics_root / args.baseline_run
    if not cb_dir.exists():
        print(f"Missing CodeBERT metrics dir: {cb_dir}")
        return 1

    cb_pred_path = cb_dir / "codebert_test_predictions.json"
    if not cb_pred_path.exists():
        cb_pred_path = cb_dir / "test_predictions.json"
    bl_clf = bl_dir / "classification_logreg.joblib"
    bl_vec = bl_dir / "classification_tfidf_vectorizer.joblib"

    baseline_recs = baseline_predictions(dataset_csv, bl_clf, bl_vec, args.baseline_run)
    if cb_pred_path.exists():
        codebert_recs = load_predictions_json(
            cb_pred_path, "codebert_finetuned", args.codebert_run, dataset_csv
        )
    else:
        print("No CodeBERT predictions file; run training first.")
        return 1

    gallery = build_gallery_entries(baseline_recs, codebert_recs)
    errors_b = [r for r in baseline_recs if not r["correct"]]
    errors_c = [r for r in codebert_recs if not r["correct"]]

    with open(cb_dir / "classification_error_gallery.json", "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2, ensure_ascii=False)
    with open(cb_dir / "error_examples.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_wrong": errors_b,
                "codebert_wrong": errors_c,
                "review_queue_note": "10 templates in review_queue.json are NOT in training CSV as accepted labels.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(cb_dir / "per_class_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": build_per_class_analysis(baseline_recs),
                "codebert": build_per_class_analysis(codebert_recs),
            },
            f,
            indent=2,
        )

    bl_metrics_path = bl_dir / "classification_metrics.json"
    cb_metrics_path = cb_dir / "codebert_classification_metrics.json"
    with open(bl_metrics_path) as f:
        bl_m = json.load(f)
    with open(cb_metrics_path) as f:
        cb_m = json.load(f)
    comparison = build_model_comparison(
        bl_m,
        cb_m,
        manifest,
        args.baseline_run,
        args.codebert_run,
        gallery["matched_test_ids"],
    )
    comparison["failure_analysis"] = {
        "baseline": failure_analysis(baseline_recs, "tfidf_logreg_baseline"),
        "codebert": failure_analysis(codebert_recs, "codebert_finetuned"),
    }
    with open(cb_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"[OK] Gallery written to {cb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
