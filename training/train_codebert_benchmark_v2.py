"""
Fine-tune CodeBERT on benchmark_v2 only (never writes to production checkpoint).

Usage:
    py -3.13 -m training.train_codebert_benchmark_v2 --epochs 4
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    data_path = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"
    if not data_path.exists():
        print("Generate data first: py -3.13 -m training.dataset_pipeline.generate_benchmark")
        return 1

    output_dir = PROJECT_ROOT / "models" / "benchmark_v2" / f"codebert_{run_id}"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_v2" / f"codebert_{run_id}"

    argv = [
        sys.executable,
        "-m",
        "training.train_codebert_classifier",
        "--data_path",
        str(data_path),
        "--output_dir",
        str(output_dir),
        "--metrics_dir",
        str(metrics_dir),
        *sys.argv[1:],
    ]
    print(" ".join(argv))
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
