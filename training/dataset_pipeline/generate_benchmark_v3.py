"""
Generate benchmark_v3 (does not modify benchmark_v2).

Usage:
    py -3.13 -m training.dataset_pipeline.generate_benchmark_v3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_VERSION = "benchmark_v3"

import training.dataset_pipeline.extra_templates  # noqa: F401
import training.dataset_pipeline.benchmark_v3_templates  # noqa: F401

from training.dataset_pipeline.generate_benchmark import (
    SEED,
    build_quality_report,
    generate_classification_records,
    write_outputs,
)
from training.dataset_pipeline.similarity_v2 import generate_similarity_pairs_v2
from training.dataset_pipeline.splits_v3 import (
    assign_family_splits_v3,
    class_coverage_by_split,
    assert_family_isolation,
    families_per_split,
)
from backend.app.config import PROJECT_ROOT as PR


def generate_v3_records(seed: int = SEED):
    accepted, rejected, review_queue = generate_classification_records(seed=seed)
    for s in accepted:
        s.pop("split", None)
    for s in accepted:
        s["dataset_version"] = DATASET_VERSION
    mapping, coverage_notes = assign_family_splits_v3(accepted, seed=seed)
    for s in accepted:
        s["split"] = mapping.get(s["family_id"], "train")
    assert_family_isolation(accepted)
    return accepted, rejected, review_queue, coverage_notes


def main() -> int:
    root = PR / "data" / DATASET_VERSION
    print(f"Generating {DATASET_VERSION} under {root}")

    samples, rejected, review_queue, cov_notes = generate_v3_records(SEED)
    pairs = generate_similarity_pairs_v2(samples, seed=SEED)
    manifest = write_outputs(root, samples, rejected, review_queue, pairs, SEED, dataset_version=DATASET_VERSION)
    manifest["split_coverage_notes"] = cov_notes
    manifest["class_coverage_by_split"] = class_coverage_by_split(samples)
    manifest["families_per_split_counts"] = {k: len(v) for k, v in families_per_split(samples).items()}
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    report = build_quality_report(samples, rejected, review_queue, pairs, manifest)
    report["dataset_version"] = DATASET_VERSION
    report["class_coverage_by_split"] = manifest["class_coverage_by_split"]
    report["split_coverage_notes"] = cov_notes
    report["synthetic_unreviewed"] = True
    report["human_reviewed"] = False
    report_path = root / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in report if k not in ("pair_leakage_errors",)}, indent=2))
    print(f"[OK] families={report['independent_program_families']} coverage={manifest['class_coverage_by_split']}")
    return 0 if report["pair_leakage_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
