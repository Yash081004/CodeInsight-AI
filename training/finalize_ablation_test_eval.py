"""
Run historical test eval once for the validation-selected ablation winner.

Usage:
    py -3.13 -m training.finalize_ablation_test_eval
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ab_root = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "ablations"
    summary_path = ab_root / "ablation_summary.json"
    if not summary_path.exists():
        print("Run: py -3.13 -m training.run_benchmark_v2_ablations")
        return 1
    summary = json.loads(summary_path.read_text())
    winner = summary["selected_by_val_macro_f1"]
    if winner == "abl_A_repro_phase3":
        print("Winner is phase3 import — test metrics already in codebert_20261019_082413.")
        out = {
            "winner": winner,
            "test_eval_source": "codebert_20261019_082413",
            "label": "historical_test_previously_inspected",
        }
        (ab_root / "final_test_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0

    models = PROJECT_ROOT / "models" / "benchmark_v2" / "ablations" / winner
    metrics = ab_root / winner
    data = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"
    cmd = [
        sys.executable,
        "-m",
        "training.train_codebert_classifier",
        "--data_path",
        str(data),
        "--output_dir",
        str(models),
        "--metrics_dir",
        str(metrics),
        "--experiment_id",
        winner,
        "--epochs",
        "0",
    ]
    # epochs 0 not supported — run test-only script inline
    import torch
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader

    from backend.app.config import ERROR_LABELS, ID_TO_LABEL
    from training.train_codebert_classifier import CodeDataset, evaluate_model

    df = pd.read_csv(data)
    df["label_id"] = df["error_label"].map({l: i for i, l in enumerate(ERROR_LABELS)})
    test_df = df[df["split"] == "test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(models))
    model = AutoModelForSequenceClassification.from_pretrained(str(models), num_labels=len(ERROR_LABELS))
    model.to(device)
    loader = DataLoader(
        CodeDataset(test_df["code"].tolist(), test_df["label_id"].tolist(), tokenizer),
        batch_size=8,
    )
    tm = evaluate_model(model, loader, device, detailed=True, return_probabilities=True)
    preds = __import__("numpy").argmax(tm["probabilities"], axis=1)
    payload = {
        "winner_experiment_id": winner,
        "eval_split": "test",
        "label": "historical_test_previously_inspected",
        "macro_f1": round(tm["macro_f1"], 4),
        "accuracy": round(tm["accuracy"], 4),
        "per_class_metrics": tm.get("per_class"),
        "note": "Single post-selection test eval; not used for tuning.",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (metrics / "historical_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ab_root / "final_test_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
