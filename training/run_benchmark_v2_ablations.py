"""
Run controlled CodeBERT ablations on benchmark_v2 (validation-only selection).

Usage:
    py -3.13 -m training.run_benchmark_v2_ablations
    py -3.13 -m training.run_benchmark_v2_ablations --only abl_D_lr3e5
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"

EXPERIMENTS = {
    "abl_A_repro_phase3": {
        "description": "Reproducibility control (phase 3 run; metrics imported, not retrained)",
        "import_from": "codebert_20261019_082413",
    },
    "abl_B_unweighted_ce": {
        "description": "Unweighted CE vs default weighted CE",
        "args": ["--no_class_weights", "--skip_test_eval", "--epochs", "2", "--batch_size", "4", "--lr", "2e-5"],
    },
    "abl_C_balanced_sampler": {
        "description": "Balanced sampler, unweighted CE",
        "args": ["--balanced_sampler", "--no_class_weights", "--skip_test_eval", "--epochs", "2", "--batch_size", "4"],
    },
    "abl_D_lr3e5_weighted": {
        "description": "Weighted CE, LR 3e-5",
        "args": ["--skip_test_eval", "--epochs", "2", "--batch_size", "4", "--lr", "3e-5"],
    },
    "abl_E_earlystop4": {
        "description": "Weighted CE, up to 4 epochs, patience 1",
        "args": ["--skip_test_eval", "--epochs", "4", "--batch_size", "4", "--patience", "1"],
    },
}


def _register_phase3_control(ablations_root: Path) -> None:
    src = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "codebert_20261019_082413"
    dst = ablations_root / "abl_A_repro_phase3"
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "abl_A_repro_phase3",
        "imported_from": "codebert_20261019_082413",
        "note": "Not retrained; phase 3 reproducibility control.",
    }
    for name in (
        "validation_metrics.json",
        "codebert_classification_metrics.json",
        "training_history.json",
        "experiment_config.json",
    ):
        sp = src / name
        if sp.exists():
            (dst / name).write_text(sp.read_text(encoding="utf-8"), encoding="utf-8")
    # Build validation_metrics from training history if missing
    vm = dst / "validation_metrics.json"
    if not vm.exists() and (src / "training_history.json").exists():
        hist = json.loads((src / "training_history.json").read_text())
        best = max(hist, key=lambda r: r.get("val_macro_f1", 0))
        json.dump(
            {
                "eval_split": "val",
                "best_epoch": best.get("epoch"),
                "macro_f1": best.get("val_macro_f1"),
                "imported": True,
            },
            vm.open("w"),
            indent=2,
        )
    (dst / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_train(exp_id: str, extra_args: list[str]) -> int:
    models = PROJECT_ROOT / "models" / "benchmark_v2" / "ablations" / exp_id
    metrics = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "ablations" / exp_id
    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "training.train_codebert_classifier",
        "--data_path",
        str(DATA),
        "--output_dir",
        str(models),
        "--metrics_dir",
        str(metrics),
        "--experiment_id",
        exp_id,
        *extra_args,
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def _summarize(ablations_root: Path) -> dict:
    rows = []
    for exp_id in sorted(EXPERIMENTS.keys()):
        mdir = ablations_root / exp_id
        vm = mdir / "validation_metrics.json"
        val_f1 = None
        collapse = None
        if vm.exists():
            v = json.loads(vm.read_text())
            val_f1 = v.get("macro_f1")
            collapse = v.get("collapse_warning")
        elif (mdir / "training_history.json").exists():
            hist = json.loads((mdir / "training_history.json").read_text())
            val_f1 = max((r.get("val_macro_f1") or 0) for r in hist)
        rows.append({"experiment_id": exp_id, "val_macro_f1": val_f1, "collapse_warning": collapse})
    best = max(rows, key=lambda r: (r["val_macro_f1"] or 0, -(r["collapse_warning"] or False)))
    tfidf_val = None
    audit = PROJECT_ROOT / "artifacts/metrics/benchmark_v2/phase4/collapse_audit.json"
    if audit.exists():
        tfidf_val = json.loads(audit.read_text()).get("tfidf_baseline_val", {}).get("macro_f1")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": rows,
        "selected_by_val_macro_f1": best["experiment_id"],
        "selection_note": "Test split not used for selection. Ties broken by lower collapse_warning when present.",
        "tfidf_baseline_val_macro_f1": tfidf_val,
        "codebert_beats_tfidf_on_val": (best["val_macro_f1"] or 0) > (tfidf_val or 0) if tfidf_val else None,
    }
    (ablations_root / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Run single experiment id")
    parser.add_argument("--cpu_matrix", action="store_true", help="Run B and D only (skip E)")
    args = parser.parse_args()

    ab_root = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / "ablations"
    ab_root.mkdir(parents=True, exist_ok=True)
    _register_phase3_control(ab_root)

    to_run = list(EXPERIMENTS.keys())
    if args.cpu_matrix:
        to_run = ["abl_B_unweighted_ce", "abl_D_lr3e5_weighted"]
    if args.only:
        to_run = [args.only]

    for exp_id in to_run:
        spec = EXPERIMENTS.get(exp_id)
        if not spec or "import_from" in spec:
            continue
        rc = _run_train(exp_id, spec["args"])
        if rc != 0:
            print(f"[FAIL] {exp_id} exit {rc}")
            return rc

    summary = _summarize(ab_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
