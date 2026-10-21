"""
Phase 7: deep error analysis for benchmark_v3 (gpu_3_balanced vs TF-IDF).

Usage:
    py -3.13 -m training.phase7_error_analysis
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID
from training.benchmark_runner import collapsed_predictions, evaluate_tfidf
from training.dataset_validation import sha256_file
from training.task_context import assess_task_context

DATASET_CSV = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
CODEBERT_EXP = "gpu_3_balanced"
VAL_PREDS = (
    PROJECT_ROOT
    / "artifacts/metrics/benchmark_v3/codebert_experiments"
    / CODEBERT_EXP
    / "validation_predictions.json"
)
MODEL_DIR = PROJECT_ROOT / "models" / "benchmark_v3" / "codebert_experiments" / CODEBERT_EXP
OUT_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "phase7"


def _load_codebert_preds(split: str) -> List[dict]:
    cache = OUT_DIR / f"codebert_{CODEBERT_EXP}_{split}_predictions.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if split == "val" and VAL_PREDS.exists():
        return json.loads(VAL_PREDS.read_text(encoding="utf-8"))

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.train_codebert_classifier import CodeDataset, evaluate_model

    df = pd.read_csv(DATASET_CSV)
    ev = df[df["split"] == split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), num_labels=len(ERROR_LABELS))
    model.to(device)
    loader = DataLoader(
        CodeDataset(
            ev["code"].tolist(),
            ev["error_label"].map(LABEL_TO_ID).tolist(),
            tokenizer,
        ),
        batch_size=8,
    )
    tm = evaluate_model(model, loader, device, detailed=True, return_probabilities=True)
    preds = np.argmax(tm["probabilities"], axis=1)
    rows = []
    for i in range(len(ev)):
        row = ev.iloc[i]
        probs = {
            ID_TO_LABEL[j]: round(float(tm["probabilities"][i][j]), 4)
            for j in range(len(ERROR_LABELS))
        }
        pl = ID_TO_LABEL[int(preds[i])]
        rows.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "source_template_id": row.get("source_template_id"),
                "true_label": row["error_label"],
                "predicted_label": pl,
                "correct": row["error_label"] == pl,
                "probabilities": probs,
                "model": "codebert_research",
                "experiment_id": CODEBERT_EXP,
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _tfidf_preds(split: str) -> List[dict]:
    cache = OUT_DIR / f"tfidf_{split}_predictions.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    df = pd.read_csv(DATASET_CSV)
    train = df[df["split"] == "train"]
    ev = df[df["split"] == split]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    y_train = train["error_label"].map(LABEL_TO_ID)
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)
    X_train = vec.fit_transform(train["code"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")
    clf.fit(X_train, y_train)
    pred = clf.predict(vec.transform(ev["code"]))
    proba = clf.predict_proba(vec.transform(ev["code"]))
    rows = []
    for i in range(len(ev)):
        row = ev.iloc[i]
        probs = {ID_TO_LABEL[j]: round(float(proba[i][j]), 4) for j in range(len(proba[0]))}
        pl = ID_TO_LABEL[int(pred[i])]
        rows.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "source_template_id": row.get("source_template_id"),
                "true_label": row["error_label"],
                "predicted_label": pl,
                "correct": row["error_label"] == pl,
                "probabilities": probs,
                "model": "tfidf_logreg",
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _prob_flatness(probs: dict) -> float:
    vals = list(probs.values())
    return float(np.std(vals)) if vals else 0.0


def _enrich(df: pd.DataFrame, preds: List[dict]) -> List[dict]:
    by_id = {p["sample_id"]: p for p in preds}
    out = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        p = by_id.get(sid, {})
        ctx = assess_task_context(
            row["error_label"],
            code=row["code"],
            expected_behavior=row.get("expected_behavior"),
            problem_type=row.get("problem_type"),
        )
        out.append(
            {
                "sample_id": sid,
                "family_id": row["family_id"],
                "source_template_id": row.get("source_template_id"),
                "split": row["split"],
                "true_label": row["error_label"],
                "predicted_label": p.get("predicted_label"),
                "correct": p.get("correct"),
                "probabilities": p.get("probabilities"),
                "probability_std": round(_prob_flatness(p.get("probabilities") or {}), 4),
                "mutation_type": row.get("mutation_type"),
                "mutation_description": row.get("mutation_description"),
                "expected_behavior": row.get("expected_behavior"),
                "validation_status": row.get("validation_status"),
                "code_excerpt": (row["code"] or "")[:400],
                "task_context": ctx,
            }
        )
    return out


def _metrics_block(entries: List[dict]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    y_true = [LABEL_TO_ID[e["true_label"]] for e in entries]
    y_pred = [LABEL_TO_ID[e["predicted_label"]] for e in entries if e.get("predicted_label") in LABEL_TO_ID]
    if len(y_pred) != len(y_true):
        return {}
    labels_idx = list(range(len(ERROR_LABELS)))
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=ERROR_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    pred_ids = np.array(y_pred)
    return {
        "per_class": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall": round(report[lbl]["recall"], 4),
                "f1": round(report[lbl]["f1-score"], 4),
                "support": int(report[lbl]["support"]),
            }
            for lbl in ERROR_LABELS
        },
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": {"labels": ERROR_LABELS, "matrix": cm.tolist()},
        "predicted_label_counts": dict(Counter(ID_TO_LABEL[int(p)] for p in pred_ids)),
        "collapse": collapsed_predictions(pred_ids),
    }


def _prioritize_review(entries_val: List[dict], cb: List[dict], tf: List[dict]) -> List[dict]:
    tf_map = {x["sample_id"]: x for x in tf}
    cb_map = {x["sample_id"]: x for x in cb}
    priority = []
    for e in entries_val:
        sid = e["sample_id"]
        tl = e["true_label"]
        both_wrong = not cb_map[sid]["correct"] and not tf_map[sid]["correct"]
        score = 0
        if tl in ("off_by_one_loop", "wrong_condition", "wrong_operator"):
            score += 3
        if both_wrong:
            score += 4
        if e["task_context"].get("requires_external_context"):
            score += 2
        if e.get("probability_std", 1) < 0.02:
            score += 2
        priority.append({**e, "review_priority_score": score, "both_models_wrong": both_wrong})
    priority.sort(key=lambda x: (-x["review_priority_score"], x["sample_id"]))
    return priority[:40]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    checksum = sha256_file(DATASET_CSV)

    cb_val = _load_codebert_preds("val")
    cb_test = _load_codebert_preds("test")
    tf_val = _tfidf_preds("val")
    tf_test = _tfidf_preds("test")

    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    cb_val_e = _enrich(val_df, cb_val)
    cb_test_e = _enrich(test_df, cb_test)
    tf_val_e = _enrich(val_df, tf_val)
    tf_test_e = _enrich(test_df, tf_test)

    # Same-label-different-logic: group by predicted label, compare code hashes
    same_pred_groups: Dict[str, List[str]] = defaultdict(list)
    for e in cb_val_e:
        if e.get("predicted_label"):
            same_pred_groups[e["predicted_label"]].append(e["sample_id"])

    flat_probs = [e for e in cb_val_e if e.get("probability_std", 1) < 0.015]
    family_errors: Dict[str, dict] = {}
    for fam, grp in val_df.groupby("family_id"):
        ids = grp["sample_id"].tolist()
        sub = [e for e in cb_val_e if e["sample_id"] in ids]
        family_errors[fam] = {
            "n_val": len(sub),
            "codebert_correct": sum(1 for e in sub if e.get("correct")),
            "true_labels": dict(Counter(grp["error_label"])),
            "predicted_labels": dict(Counter(e.get("predicted_label") for e in sub)),
        }

    both_wrong_val = [
        e["sample_id"]
        for e in cb_val_e
        if not next(x for x in cb_val if x["sample_id"] == e["sample_id"])["correct"]
        and not next(x for x in tf_val if x["sample_id"] == e["sample_id"])["correct"]
    ]

    hypotheses = []
    verified = []
    if len(flat_probs) >= 5:
        verified.append(
            {
                "finding": "codebert_near_uniform_softmax_on_val",
                "evidence": f"{len(flat_probs)} val samples with probability std < 0.015",
                "example_ids": [e["sample_id"] for e in flat_probs[:8]],
                "artifact": str(VAL_PREDS.relative_to(PROJECT_ROOT)),
                "interpretation": "Verified: logits barely separate classes on these snippets (not proof of lexical-only features, but weak discriminative signal).",
            }
        )
    if family_errors.get("factorial", {}).get("predicted_labels", {}).get("wrong_initialization", 0) >= 3:
        verified.append(
            {
                "finding": "family_factorial_codebert_favors_wrong_initialization",
                "evidence": family_errors["factorial"],
                "example_ids": [e["sample_id"] for e in cb_val_e if e["family_id"] == "factorial"][:6],
                "artifact": "artifacts/metrics/benchmark_v3/codebert_experiments/gpu_3_balanced/validation_predictions.json",
            }
        )
    hypotheses.append(
        {
            "hypothesis": "balanced_sampler_inflates_wrong_initialization_predictions",
            "status": "consistent_with_evidence_not_proven_causal",
            "evidence": "gpu_3_balanced val predicted_label_counts: wrong_initialization=14/30; training uses oversampling of minority fault labels.",
            "artifact": "artifacts/metrics/benchmark_v3/codebert_experiments/gpu_3_balanced/validation_metrics.json",
        }
    )

    review_batch = _prioritize_review(cb_val_e, cb_val, tf_val)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "dataset_version": "benchmark_v3",
        "dataset_checksum_sha256": checksum,
        "codebert_experiment_id": CODEBERT_EXP,
        "label_quality": "synthetic_unreviewed",
        "verified_facts": verified,
        "hypotheses": hypotheses,
        "unresolved_questions": [
            "Whether template/family naming tokens correlate with TF-IDF decisions (not measured in this report).",
            "Whether human adjudication would change off_by_one_loop / wrong_condition labels for factorial-heavy val set.",
        ],
        "validation": {
            "codebert": _metrics_block(cb_val_e),
            "tfidf": _metrics_block(tf_val_e),
        },
        "test_historical": {
            "codebert": _metrics_block(cb_test_e),
            "tfidf": _metrics_block(tf_test_e),
            "policy_note": "Test inspected post Phase 6; not used for tuning in Phase 7.",
        },
        "family_summary_val_codebert": family_errors,
        "both_models_wrong_val_sample_ids": both_wrong_val,
        "near_uniform_softmax_val_sample_ids": [e["sample_id"] for e in flat_probs],
        "ambiguous_or_context_dependent_val": [
            {
                "sample_id": e["sample_id"],
                "true_label": e["true_label"],
                "reasons": e["task_context"].get("limitations", []),
            }
            for e in cb_val_e
            if e["task_context"].get("requires_external_context")
        ],
        "review_batch_preview": [
            {
                "sample_id": r["sample_id"],
                "family_id": r["family_id"],
                "true_label": r["true_label"],
                "review_priority_score": r["review_priority_score"],
                "both_models_wrong": r["both_models_wrong"],
            }
            for r in review_batch[:20]
        ],
    }

    entries_path = OUT_DIR / "error_explorer_entries.json"
    entries_path.write_text(
        json.dumps(
            {
                "val": {"codebert": cb_val_e, "tfidf": tf_val_e},
                "test": {"codebert": cb_test_e, "tfidf": tf_test_e},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "error_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Review decisions scaffold (append-only; all pending)
    review_path = PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json"
    if not review_path.exists():
        decisions = []
        for i, row in enumerate(review_batch, start=1):
            decisions.append(
                {
                    "review_id": f"V3RQ{i:03d}",
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "source_template_id": row.get("source_template_id"),
                    "current_label": row["true_label"],
                    "original_label": row["true_label"],
                    "status": "pending",
                    "reviewer_id": None,
                    "rationale": None,
                    "code_ref": f"data/benchmark_v3/classification/dataset.csv#{row['sample_id']}",
                    "expected_behavior": row.get("expected_behavior"),
                    "mutation_type": row.get("mutation_type"),
                    "mutation_rationale": row.get("mutation_description"),
                    "behavioral_tests_note": "Template finite tests only; see validation_status on sample.",
                    "priority_score": row["review_priority_score"],
                }
            )
        review_path.write_text(
            json.dumps(
                {
                    "dataset_version": "benchmark_v3",
                    "human_reviewed": False,
                    "adjudication_complete": False,
                    "decisions": decisions,
                    "note": "Decisions append audit history; dataset.csv labels are not overwritten by this file.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    reviewed_manifest = PROJECT_ROOT / "data" / "benchmark_reviewed" / "manifest.json"
    reviewed_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not reviewed_manifest.exists():
        reviewed_manifest.write_text(
            json.dumps(
                {
                    "status": "blocked_pending_human_review",
                    "human_reviewed_sample_count": 0,
                    "requirements": [
                        "No family overlap with benchmark_v3 train split",
                        "Explicit intended behavior per sample",
                        "Reviewer status accepted or relabeled with rationale",
                        "All five labels where feasible",
                    ],
                    "eval_slice_path": None,
                    "note": "Credible reviewed-set evaluation blocked until at least one accepted adjudication is exported.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"ok": True, "report": str(OUT_DIR / "error_analysis_report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
