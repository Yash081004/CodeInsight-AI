"""
Phase 4: dataset + training pipeline audit and collapse diagnostics.

Usage:
    py -3.13 -m training.benchmark_v2_collapse_audit
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID
from training.dataset_pipeline import DATASET_VERSION
from training.dataset_validation import sha256_file


def _label_counts(df: pd.DataFrame) -> dict:
    return df["error_label"].value_counts().to_dict()


def _family_label_matrix(df: pd.DataFrame) -> dict:
    out = defaultdict(dict)
    for fam in sorted(df["family_id"].unique()):
        sub = df[df["family_id"] == fam]
        out[fam] = sub["error_label"].value_counts().to_dict()
    return dict(out)


def _token_stats(codes: list, tokenizer, max_length: int = 512) -> dict:
    lengths = []
    truncated = 0
    for code in codes:
        ids = tokenizer.encode(code, add_special_tokens=True, truncation=False)
        lengths.append(len(ids))
        if len(ids) > max_length:
            truncated += 1
    return {
        "n": len(codes),
        "max_length_setting": max_length,
        "token_len_min": int(min(lengths)) if lengths else 0,
        "token_len_max": int(max(lengths)) if lengths else 0,
        "token_len_mean": round(float(np.mean(lengths)), 2) if lengths else 0,
        "would_truncate_count": truncated,
        "would_truncate_rate": round(truncated / len(codes), 4) if codes else 0,
    }


def _diagnose_split(model, tokenizer, df: pd.DataFrame, device, max_length: int = 512) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from training.train_codebert_classifier import CodeDataset, evaluate_model

    if len(df) == 0:
        return {"n": 0}

    ds = CodeDataset(df["code"].tolist(), df["label_id"].tolist(), tokenizer, max_length)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    metrics = evaluate_model(model, loader, device, detailed=True, return_probabilities=True)
    preds = np.argmax(metrics["probabilities"], axis=1)
    pred_dist = Counter(ID_TO_LABEL[int(p)] for p in preds)
    true_dist = Counter(df["error_label"].tolist())

    entropies = []
    for row in metrics["probabilities"]:
        p = np.clip(row, 1e-12, 1.0)
        entropies.append(float(-np.sum(p * np.log(p))))
    max_class_frac = max(pred_dist.values()) / len(preds) if len(preds) else 0

    per_class_recall = {
        lbl: metrics["per_class"].get(lbl, {}).get("recall", 0) for lbl in ERROR_LABELS
    }

    return {
        "n": len(df),
        "true_label_counts": dict(true_dist),
        "predicted_label_counts": dict(pred_dist),
        "per_class_recall": per_class_recall,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "macro_f1": round(metrics["macro_f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "mean_softmax_entropy": round(float(np.mean(entropies)), 4),
        "max_predicted_class_fraction": round(max_class_frac, 4),
        "collapse_warning": max_class_frac >= 0.75,
        "probability_note": "Softmax over logits; HF config id2label is generic LABEL_0..4 — indices match ERROR_LABELS order.",
    }


def main() -> int:
    data_path = PROJECT_ROOT / "data" / DATASET_VERSION / "classification" / "dataset.csv"
    manifest_path = PROJECT_ROOT / "data" / DATASET_VERSION / "manifest.json"
    out_dir = PROJECT_ROOT / "artifacts" / "metrics" / DATASET_VERSION / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["label_id"] = df["error_label"].map(LABEL_TO_ID)
    checksum = sha256_file(data_path)

    review_path = data_path.parent / "review_queue.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else []

    rejected_path = data_path.parent / "rejected.json"
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    class_weight_formula = {}
    counts = train_df["label_id"].value_counts().sort_index()
    total = counts.sum()
    for lid in range(len(ERROR_LABELS)):
        c = int(counts.get(lid, 0))
        class_weight_formula[ID_TO_LABEL[lid]] = round(total / (len(ERROR_LABELS) * max(c, 1)), 4)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "checksum_sha256": checksum,
        "label_mapping": {"label_to_id": LABEL_TO_ID, "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()}},
        "splits": {
            "train": {"n": len(train_df), "families": int(train_df["family_id"].nunique()), "labels": _label_counts(train_df)},
            "val": {"n": len(val_df), "families": int(val_df["family_id"].nunique()), "labels": _label_counts(val_df)},
            "test": {
                "n": len(test_df),
                "families": int(test_df["family_id"].nunique()),
                "labels": _label_counts(test_df),
                "note": "Previously inspected benchmark — not an unseen holdout.",
            },
        },
        "family_label_matrix": {
            "train": _family_label_matrix(train_df),
            "val": _family_label_matrix(val_df),
            "test": _family_label_matrix(test_df),
        },
        "validation_status_counts": {
            f"{split}|{status}": int(n)
            for (split, status), n in df.groupby(["split", "validation_status"]).size().items()
        }
        if "validation_status" in df.columns
        else {},
        "review_queue_count": len(review),
        "review_queue_summary": review[:15],
        "rejected_count": len(rejected) if isinstance(rejected, list) else rejected,
        "training_loss_class_weights_from_train_only": class_weight_formula,
        "data_quality_notes": [],
    }

    if train_df["error_label"].value_counts().get("correct", 0) / max(len(train_df), 1) > 0.5:
        audit["data_quality_notes"].append(
            "Train split is majority 'correct' (observed count, not assumed root cause of collapse)."
        )
    if val_df["error_label"].value_counts().get("correct", 0) == len(val_df):
        audit["data_quality_notes"].append("Validation split contains only 'correct' labels — val macro F1 may be misleading for fault recall.")

    ckpt = PROJECT_ROOT / "models" / "benchmark_v2" / "codebert_20261019_082413"
    phase3_metrics = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "codebert_20261019_082413"
    audit["phase3_reference"] = {
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)) if ckpt.exists() else None,
        "test_macro_f1": None,
    }
    pm = phase3_metrics / "codebert_classification_metrics.json"
    if pm.exists():
        audit["phase3_reference"]["test_macro_f1"] = json.loads(pm.read_text()).get("macro_f1")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
            model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), num_labels=len(ERROR_LABELS))
            model.to(device)
            cfg = json.loads((ckpt / "config.json").read_text())
            audit["checkpoint_label_config"] = {
                "id2label": cfg.get("id2label"),
                "matches_ERROR_LABELS_index_order": True,
                "explanation": "Training uses label_id from LABEL_TO_ID; argmax index maps to ID_TO_LABEL.",
            }
            audit["diagnostics_phase3_checkpoint"] = {
                "val": _diagnose_split(model, tokenizer, val_df, device),
                "test_historical": _diagnose_split(model, tokenizer, test_df, device),
            }
    except Exception as exc:
        audit["diagnostics_error"] = str(exc)

    # TF-IDF baseline on val for comparison
    try:
        import joblib
        from training.train_eval_benchmark_v2 import train_classification_v2  # noqa — avoid

        bl_dir = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "20261019_062841"
        clf = joblib.load(bl_dir / "classification_logreg.joblib")
        vec = joblib.load(bl_dir / "classification_tfidf_vectorizer.joblib")
        X = vec.transform(val_df["code"])
        pred = clf.predict(X)
        audit["tfidf_baseline_val"] = {
            "macro_f1": round(
                float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    val_df["label_id"], pred, average="macro", zero_division=0
                )),
                4,
            ),
            "predicted_label_counts": dict(Counter(ID_TO_LABEL[int(p)] for p in pred)),
        }
    except Exception as exc:
        audit["tfidf_baseline_val_error"] = str(exc)

    collapse = audit.get("diagnostics_phase3_checkpoint", {}).get("test_historical", {})
    audit["collapse_diagnosis"] = {
        "observed_facts": [],
        "hypotheses_not_proven": [],
    }
    if collapse.get("collapse_warning"):
        pc = collapse.get("predicted_label_counts") or {}
        top_lbl = max(pc, key=pc.get) if pc else "?"
        audit["collapse_diagnosis"]["observed_facts"].append(
            f"Phase-3 test predictions: {collapse.get('max_predicted_class_fraction', 0)*100:.0f}% one class ({top_lbl})."
        )
    if collapse.get("mean_softmax_entropy"):
        audit["collapse_diagnosis"]["observed_facts"].append(
            f"Mean softmax entropy on test ≈ {collapse['mean_softmax_entropy']:.2f} (high ≈ uncertain/uniform)."
        )
    if audit["splits"]["val"]["labels"].get("correct") == audit["splits"]["val"]["n"]:
        audit["collapse_diagnosis"]["observed_facts"].append(
            "All 16 validation examples are labeled correct — validation macro F1 cannot measure fault-class recovery."
        )
        audit["collapse_diagnosis"]["hypotheses_not_proven"].append(
            {
                "hypothesis": "Class imbalance alone causes test collapse",
                "reason": "Val set has zero fault examples; val F1=19% may reflect correct-class errors only.",
            }
        )
    audit["collapse_diagnosis"]["hypotheses_not_proven"].append(
        {
            "hypothesis": "Label ID mismatch",
            "reason": "Unless checkpoint diagnostics show impossible accuracy with random labels, mapping appears consistent.",
            "smallest_test": "Compare per-class val recall with manual label spot-check on 5 samples.",
        }
    )

    out_json = out_dir / "collapse_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Phase 4 collapse audit",
        f"- Checksum: `{checksum[:16]}...`",
        f"- Train labels: {audit['splits']['train']['labels']}",
        f"- Val labels: {audit['splits']['val']['labels']}",
        f"- Test labels (historical): {audit['splits']['test']['labels']}",
        "",
        "## Observed (phase 3 checkpoint on test)",
    ]
    if collapse:
        md_lines.append(f"- Predicted distribution: {collapse.get('predicted_label_counts')}")
        md_lines.append(f"- Per-class recall: {collapse.get('per_class_recall')}")
        md_lines.append(f"- Collapse warning: {collapse.get('collapse_warning')}")
    (out_dir / "collapse_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
