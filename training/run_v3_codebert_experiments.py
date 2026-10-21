"""
Controlled CodeBERT experiments on benchmark_v3 (validation-only selection).

Usage:
    py -3.13 -m training.run_v3_codebert_experiments --preflight-only
    py -3.13 -m training.run_v3_codebert_experiments --allow-cpu --epochs 2
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, ID_TO_LABEL
from training.benchmark_runner import collapsed_predictions, evaluate_tfidf
from training.dataset_pipeline.splits import verify_group_isolation
from training.dataset_validation import sha256_file

DATASET = "benchmark_v3"
DATA_CSV = PROJECT_ROOT / "data" / DATASET / "classification" / "dataset.csv"
MANIFEST = PROJECT_ROOT / "data" / DATASET / "manifest.json"
EXPECTED_CHECKSUM = None  # loaded from manifest

EXPERIMENTS = {
    "gpu_1_baseline": {
        "lr": "2e-5",
        "args": ["--lr", "2e-5", "--skip_test_eval"],
        "class_weights": True,
        "balanced_sampler": False,
    },
    "gpu_2_lr3e5": {
        "lr": "3e-5",
        "args": ["--lr", "3e-5", "--skip_test_eval"],
        "class_weights": True,
        "balanced_sampler": False,
    },
    "gpu_3_balanced": {
        "lr": "2e-5",
        "args": ["--lr", "2e-5", "--balanced_sampler", "--no_class_weights", "--skip_test_eval"],
        "class_weights": False,
        "balanced_sampler": True,
    },
}


def preflight() -> tuple[bool, dict]:
    report: dict = {
        "dataset": DATASET,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "errors": [],
    }
    if not DATA_CSV.exists():
        report["errors"].append("missing_v3_dataset")
        return False, report
    checksum = sha256_file(DATA_CSV)
    report["checksum_sha256"] = checksum
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        report["split_manifest"] = manifest.get("families_per_split")
        report["class_coverage"] = manifest.get("class_coverage_by_split")
        if manifest.get("checksum_sha256") and manifest["checksum_sha256"] != checksum:
            report["errors"].append("checksum_manifest_mismatch")
    df = pd.read_csv(DATA_CSV)
    ok, iso = verify_group_isolation(df.to_dict("records"), "family_id")
    if not ok:
        report["errors"].extend(iso[:5])
    invalid = set(df["error_label"]) - set(ERROR_LABELS)
    if invalid:
        report["errors"].append(f"invalid_labels={invalid}")
    tfidf_path = PROJECT_ROOT / "artifacts/metrics/benchmark_v3/runner/tfidf_val_metrics.json"
    if tfidf_path.exists():
        report["tfidf_baseline_val"] = json.loads(tfidf_path.read_text(encoding="utf-8"))
    else:
        report["tfidf_baseline_val"] = evaluate_tfidf(df, "val")
    report["label_quality"] = "synthetic_unreviewed"
    return len(report["errors"]) == 0, report


def _run_one(exp_id: str, spec: dict, epochs: int, patience: int, seed: int, allow_cpu: bool) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        return {"experiment_id": exp_id, "skipped": True, "reason": "no_gpu_and_allow_cpu_false"}

    models = PROJECT_ROOT / "models" / DATASET / "codebert_experiments" / exp_id
    metrics = PROJECT_ROOT / "artifacts" / "metrics" / DATASET / "codebert_experiments" / exp_id
    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "training.train_codebert_classifier",
        "--data_path",
        str(DATA_CSV),
        "--output_dir",
        str(models),
        "--metrics_dir",
        str(metrics),
        "--experiment_id",
        exp_id,
        "--epochs",
        str(epochs),
        "--batch_size",
        "8" if device == "cuda" else "4",
        "--eval_batch_size",
        "16" if device == "cuda" else "8",
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        *spec["args"],
    ]
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0

    run_meta = {
        "experiment_id": exp_id,
        "exit_code": rc,
        "device_used": device,
        "gpu_run": device == "cuda",
        "elapsed_seconds": round(elapsed, 1),
        "checkpoint_dir": str(models.relative_to(PROJECT_ROOT)),
        "metrics_dir": str(metrics.relative_to(PROJECT_ROOT)),
        "configuration": {
            "lr": spec["lr"],
            "class_weights": spec["class_weights"],
            "balanced_sampler": spec["balanced_sampler"],
            "epochs_requested": epochs,
            "patience": patience,
            "seed": seed,
        },
    }
    vm = metrics / "validation_metrics.json"
    if vm.exists():
        run_meta["validation_metrics"] = json.loads(vm.read_text(encoding="utf-8"))
    hist = metrics / "training_history.json"
    if not hist.exists():
        hist = models / "training_history.json"
    if hist.exists():
        run_meta["training_history"] = json.loads(hist.read_text(encoding="utf-8"))
    with open(metrics / "experiment_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    return run_meta


def select_candidate(results: list[dict], tfidf_macro_f1: float) -> dict:
    ok_runs = [r for r in results if r.get("exit_code") == 0 and r.get("validation_metrics")]
    if not ok_runs:
        return {"selected": None, "reason": "no_successful_runs"}

    def score(r):
        vm = r["validation_metrics"]
        macro = vm.get("macro_f1") or 0
        collapse = vm.get("collapse_warning", True)
        fault_recall = sum(
            (vm.get("per_class_metrics") or {}).get(lbl, {}).get("recall") or 0
            for lbl in ERROR_LABELS
            if lbl != "correct"
        )
        return (macro, fault_recall, 0 if collapse else 1)

    best = max(ok_runs, key=score)
    vm = best["validation_metrics"]
    beats_tfidf = (vm.get("macro_f1") or 0) > tfidf_macro_f1
    return {
        "selected_experiment_id": best["experiment_id"],
        "validation_macro_f1": vm.get("macro_f1"),
        "validation_collapse_warning": vm.get("collapse_warning"),
        "predicted_label_counts_val": vm.get("predicted_label_counts"),
        "per_class_recall_val": {
            lbl: (vm.get("per_class_metrics") or {}).get(lbl, {}).get("recall") for lbl in ERROR_LABELS
        },
        "beats_tfidf_val_macro_f1": beats_tfidf,
        "tfidf_val_macro_f1": tfidf_macro_f1,
        "selection_rule": "validation_macro_f1 then fault recall sum then non-collapse",
        "note": "Test split not used for selection.",
    }


def final_test_eval(exp_id: str) -> dict:
    """Single post-selection test eval (synthetic labels — unreviewed)."""
    import numpy as np
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = PROJECT_ROOT / "models" / DATASET / "codebert_experiments" / exp_id
    metrics = PROJECT_ROOT / "artifacts" / "metrics" / DATASET / "codebert_experiments" / exp_id
    df = pd.read_csv(DATA_CSV)
    test_df = df[df["split"] == "test"]
    from backend.app.config import LABEL_TO_ID
    from training.train_codebert_classifier import CodeDataset, evaluate_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(models))
    model = AutoModelForSequenceClassification.from_pretrained(str(models), num_labels=len(ERROR_LABELS))
    model.to(device)
    loader = DataLoader(
        CodeDataset(
            test_df["code"].tolist(),
            test_df["error_label"].map(LABEL_TO_ID).tolist(),
            tokenizer,
        ),
        batch_size=8,
    )
    tm = evaluate_model(model, loader, device, detailed=True, return_probabilities=True)
    preds = np.argmax(tm["probabilities"], axis=1)
    payload = {
        "eval_split": "test",
        "dataset_version": DATASET,
        "label_quality": "synthetic_unreviewed_not_human_verified",
        "experiment_id": exp_id,
        "policy": "single_post_selection_historical_style_eval_do_not_retune",
        "macro_f1": round(tm["macro_f1"], 4),
        "accuracy": round(tm["accuracy"], 4),
        "per_class_metrics": tm.get("per_class"),
        "collapse": collapsed_predictions(preds),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (metrics / "final_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Required when CUDA unavailable")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-test", action="store_true", help="Run once on test after selection")
    parser.add_argument("--skip-train", action="store_true", help="Only summarize existing runs")
    args = parser.parse_args()

    ok, pre = preflight()
    out_root = PROJECT_ROOT / "artifacts/metrics/benchmark_v3/codebert_experiments"
    out_root.mkdir(parents=True, exist_ok=True)
    pre_path = out_root / "preflight.json"
    pre_path.write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(json.dumps(pre, indent=2))

    if not ok:
        print("[FAIL] preflight")
        return 1
    if args.preflight_only:
        if not torch.cuda.is_available():
            print("[INFO] CUDA not available — use --allow-cpu to run on CPU (slow, logged explicitly).")
        return 0

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("[STOP] No GPU. Refusing to train without --allow-cpu.")
        return 2

    tfidf_f1 = (pre.get("tfidf_baseline_val") or {}).get("macro_f1", 0)

    results = []
    if not args.skip_train:
        for exp_id, spec in EXPERIMENTS.items():
            print(f"\n=== {exp_id} ===")
            results.append(_run_one(exp_id, spec, args.epochs, args.patience, args.seed, args.allow_cpu))

    if args.skip_train:
        for exp_id in EXPERIMENTS:
            mp = out_root / exp_id / "experiment_run.json"
            if mp.exists():
                results.append(json.loads(mp.read_text(encoding="utf-8")))

    selection = select_candidate(results, tfidf_f1)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device_policy": "cuda_preferred",
        "any_gpu_run": any(r.get("gpu_run") for r in results),
        "runs": results,
        "selection": selection,
        "label_quality_warning": "synthetic_unreviewed",
    }
    if args.final_test and selection.get("selected_experiment_id"):
        summary["final_test"] = final_test_eval(selection["selected_experiment_id"])
    (out_root / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
