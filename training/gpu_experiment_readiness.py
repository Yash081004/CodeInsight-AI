"""
Check benchmark_v3 integrity and emit GPU experiment matrix (does not train).

Usage:
    py -3.13 -m training.gpu_experiment_readiness
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    v3_csv = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    ok = v3_csv.exists()
    phase5 = PROJECT_ROOT / "artifacts/metrics/phase5/audit_report.json"
    ms = None
    if (PROJECT_ROOT / "artifacts/metrics/benchmark_v2/codebert_20261019_082413/codebert_classification_metrics.json").exists():
        ms = json.loads(
            (
                PROJECT_ROOT / "artifacts/metrics/benchmark_v2/codebert_20261019_082413/codebert_classification_metrics.json"
            ).read_text()
        ).get("inference_ms_per_sample_test_cpu")

    matrix = [
        {"id": "gpu_1_baseline", "lr": "2e-5", "epochs": 4, "class_weights": True, "balanced_sampler": False},
        {"id": "gpu_2_lr3e5", "lr": "3e-5", "epochs": 4, "class_weights": True, "balanced_sampler": False},
        {"id": "gpu_3_balanced", "lr": "2e-5", "epochs": 4, "class_weights": False, "balanced_sampler": True},
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_v3_dataset_present": ok,
        "phase5_audit_present": phase5.exists(),
        "selection_metric": "validation_macro_f1",
        "measured_cpu_ms_per_sample_phase3": ms,
        "estimated_gpu_note": "Multiply CPU epoch time by ~0.05–0.2 if GPU available; measure locally — not fabricated.",
        "proposed_matrix": matrix,
        "do_not_run_automatically": True,
    }
    out = PROJECT_ROOT / "artifacts/metrics/benchmark_v3/gpu_readiness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
