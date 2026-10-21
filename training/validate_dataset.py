"""
CLI: validate classification and similarity datasets; write reports.

Usage:
    py -3.13 -m training.validate_dataset
    py -3.13 -m training.validate_dataset --no-behavioral
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from training.dataset_validation import (
    save_dataset_metadata,
    validate_classification_dataset,
    validate_similarity_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CodeInsight AI datasets")
    parser.add_argument(
        "--no-behavioral",
        action="store_true",
        help="Skip subprocess behavioral re-checks (faster)",
    )
    args = parser.parse_args()

    settings = get_settings()
    clf_path = settings.classification_data_path / "dataset.csv"
    pairs_path = settings.similarity_data_path / "pairs.csv"
    metrics_dir = settings.artifacts_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Classification dataset validation")
    print("=" * 60)
    clf_report = validate_classification_dataset(
        clf_path,
        run_behavioral_checks=not args.no_behavioral,
    )
    print(json.dumps({k: v for k, v in clf_report.items() if k != "issues"}, indent=2))
    if clf_report.get("issues"):
        print(f"Issues: {len(clf_report['issues'])} (see report JSON)")
    print(f"Valid: {clf_report.get('valid')}")

    print("\n" + "=" * 60)
    print("Similarity pairs validation")
    print("=" * 60)
    sim_report = validate_similarity_pairs(pairs_path, clf_path)
    print(json.dumps({k: v for k, v in sim_report.items() if k != "issues"}, indent=2))
    if sim_report.get("issues"):
        print(f"Issues: {len(sim_report['issues'])} (see report JSON)")
    print(f"Valid: {sim_report.get('valid')}")

    with open(metrics_dir / "classification_dataset_validation.json", "w") as f:
        json.dump(clf_report, f, indent=2)
    with open(metrics_dir / "similarity_dataset_validation.json", "w") as f:
        json.dump(sim_report, f, indent=2)

    meta_path = save_dataset_metadata(
        clf_report,
        sim_report,
        settings.classification_data_path,
    )
    print(f"\n[OK] dataset_metadata.json -> {meta_path}")
    print(f"[OK] validation reports -> {metrics_dir}")

    return 0 if clf_report.get("valid") and sim_report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
