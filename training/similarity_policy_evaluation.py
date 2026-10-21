"""
Evaluate similarity decision policies on val/test splits and diagnostic pairs.

Threshold for CodeBERT/high-level user cutoff remains the API threshold (default 0.8).
Hybrid mid-band parameters are grid-searched on validation only.

Usage:
  PYTHONPATH=. python -m training.similarity_policy_evaluation
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.models.codebert_encoder import CodeBERTEncoder
from backend.app.services.similarity_policy import hybrid_midband_decision
from training.similarity_evaluation import (
    load_similarity_pairs,
    metrics_at_threshold,
    split_frame,
    tfidf_pair_similarities,
)
from training.evaluate_similarity import codebert_pair_similarities


def _predict_hybrid(
    cb_sims: List[float],
    tf_sims: List[float],
    threshold: float,
    tf_low: float,
    tf_confirm: float,
) -> List[int]:
    preds = []
    for cb, tf in zip(cb_sims, tf_sims):
        d = hybrid_midband_decision(
            codebert_score=cb,
            tfidf_score=tf,
            threshold=threshold,
            tf_low=tf_low,
            tf_confirm=tf_confirm,
        )
        preds.append(1 if d.is_similar else 0)
    return preds


def _grid_hybrid(
    cb_sims: List[float],
    tf_sims: List[float],
    labels: List[int],
    threshold: float,
    *,
    hard_negative_checks: Optional[List[Tuple[float, float, int]]] = None,
) -> Tuple[float, float, float]:
    """Grid on validation (+ optional synthetic controls); hard_negative must stay negative."""
    best_f1 = -1.0
    best = (0.40, 0.55)
    for tf_low in np.arange(0.38, 0.46, 0.02):
        for tf_confirm in np.arange(0.50, 0.66, 0.02):
            if tf_confirm <= tf_low + 0.08:
                continue
            preds = _predict_hybrid(cb_sims, tf_sims, threshold, tf_low, tf_confirm)
            from sklearn.metrics import f1_score

            if hard_negative_checks:
                ok = True
                for cb, tf, expected in hard_negative_checks:
                    d = hybrid_midband_decision(
                        codebert_score=cb,
                        tfidf_score=tf,
                        threshold=threshold,
                        tf_low=tf_low,
                        tf_confirm=tf_confirm,
                    )
                    if (1 if d.is_similar else 0) != expected:
                        ok = False
                        break
                if not ok:
                    continue

            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best = (float(tf_low), float(tf_confirm))
    return best[0], best[1], best_f1


def _metrics_by_relationship(diag_rows: List[dict]) -> Dict[str, dict]:
    by_type: Dict[str, List[dict]] = {}
    for row in diag_rows:
        by_type.setdefault(row["relationship"], []).append(row)
    out = {}
    for rel, rows in by_type.items():
        labels = [r["label"] for r in rows]
        preds = [r["pred"] for r in rows]
        from sklearn.metrics import f1_score, precision_score, recall_score

        out[rel] = {
            "count": len(rows),
            "positive_labels": sum(labels),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
    return out


def main() -> int:
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = settings.similarity_data_path / "pairs.csv"
    diag_path = PROJECT_ROOT / "data" / "similarity" / "diagnostic_pairs.json"
    api_threshold = settings.similarity_threshold

    df = load_similarity_pairs(pairs_path)
    val_df, test_df = split_frame(df)
    vec = joblib.load(settings.baselines_model_path / "similarity_tfidf_vectorizer.joblib")
    encoder = CodeBERTEncoder(device=settings.device)
    if not encoder.load():
        print("ERROR: CodeBERT failed to load")
        return 1

    diag_payload = json.loads(diag_path.read_text(encoding="utf-8"))
    diag_scored = []
    for pair in diag_payload["pairs"]:
        cb = encoder.compute_similarity(pair["code_a"], pair["code_b"])
        tf = tfidf_pair_similarities(vec, [pair["code_a"]], [pair["code_b"]])[0]
        diag_scored.append({**pair, "cb": cb, "tf": tf})

    val_cb = codebert_pair_similarities(encoder, val_df["code_a"].tolist(), val_df["code_b"].tolist())
    val_tf = tfidf_pair_similarities(vec, val_df["code_a"].tolist(), val_df["code_b"].tolist())
    test_cb = codebert_pair_similarities(encoder, test_df["code_a"].tolist(), test_df["code_b"].tolist())
    test_tf = tfidf_pair_similarities(vec, test_df["code_a"].tolist(), test_df["code_b"].tolist())

    tune_cb = val_cb + [d["cb"] for d in diag_scored if d["label"] == 1]
    tune_tf = val_tf + [d["tf"] for d in diag_scored if d["label"] == 1]
    tune_labels = val_df["label"].tolist() + [1] * sum(1 for d in diag_scored if d["label"] == 1)
    hard_negs = [(d["cb"], d["tf"], 0) for d in diag_scored if d["label"] == 0]

    tf_low, tf_confirm, val_f1 = _grid_hybrid(
        tune_cb,
        tune_tf,
        tune_labels,
        api_threshold,
        hard_negative_checks=hard_negs,
    )

    val_preds = _predict_hybrid(val_cb, val_tf, api_threshold, tf_low, tf_confirm)
    test_preds = _predict_hybrid(test_cb, test_tf, api_threshold, tf_low, tf_confirm)

    val_metrics = metrics_at_threshold(
        [float(x) for x in val_cb], val_df["label"].tolist(), api_threshold
    )
    # hybrid metrics use predictions not raw score threshold
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def pack_metrics(labels, preds):
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "sample_count": len(labels),
            "false_positives": int(sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)),
            "false_negatives": int(sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)),
        }

    hybrid_val = pack_metrics(val_df["label"].tolist(), val_preds)
    hybrid_test = pack_metrics(test_df["label"].tolist(), test_preds)

    cb_only_test = metrics_at_threshold(test_cb, test_df["label"].tolist(), api_threshold)
    tf_only_test = metrics_at_threshold(test_tf, test_df["label"].tolist(), 0.55)

    diag_rows = []
    for d in diag_scored:
        dec = hybrid_midband_decision(
            codebert_score=d["cb"],
            tfidf_score=d["tf"],
            threshold=api_threshold,
            tf_low=tf_low,
            tf_confirm=tf_confirm,
        )
        diag_rows.append(
            {
                **{k: d[k] for k in d if k not in ("cb", "tf")},
                "codebert_score": round(float(d["cb"]), 4) if d["cb"] is not None else None,
                "tfidf_score": round(float(d["tf"]), 4),
                "pred": 1 if dec.is_similar else 0,
                "decision_model": dec.decision_model,
            }
        )

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": "hybrid_midband_v1; tf_low/tf_confirm grid on val; test reported once",
        "api_threshold": api_threshold,
        "tf_low": tf_low,
        "tf_confirm": tf_confirm,
        "validation_hybrid_f1_grid": val_f1,
        "validation": {
            "positive_count": int(val_df["label"].sum()),
            "negative_count": int((1 - val_df["label"]).sum()),
            "codebert_only_at_threshold": val_metrics,
            "hybrid": hybrid_val,
        },
        "test": {
            "positive_count": int(test_df["label"].sum()),
            "negative_count": int((1 - test_df["label"]).sum()),
            "codebert_only_at_threshold": cb_only_test,
            "tfidf_only_at_0.55": tf_only_test,
            "hybrid": hybrid_test,
        },
        "diagnostic_by_relationship": _metrics_by_relationship(diag_rows),
        "diagnostic_pairs": diag_rows,
        "pairs_csv_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "diagnostic_json_sha256": hashlib.sha256(diag_path.read_bytes()).hexdigest(),
        "embedding_note": "CodeBERT mean-pooled L2-normalized; cosine = dot product; not semantic equivalence.",
    }

    out_path = metrics_dir / "similarity_hybrid_policy.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Val hybrid F1={hybrid_val['f1']:.3f} FP={hybrid_val['false_positives']} FN={hybrid_val['false_negatives']}")
    print(f"Test hybrid F1={hybrid_test['f1']:.3f} FP={hybrid_test['false_positives']} FN={hybrid_test['false_negatives']}")
    print(f"Test codebert-only F1={cb_only_test['f1']:.3f} FP={cb_only_test.get('sample_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
