"""
Generate benchmark_v2 classification + similarity datasets.

Usage:
    py -3.13 -m training.dataset_pipeline.generate_benchmark
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.app.config import ERROR_LABELS, PROJECT_ROOT
from training.dataset_pipeline import DATASET_VERSION
from training.dataset_pipeline.similarity_v2 import generate_similarity_pairs_v2, code_hash
from training.dataset_pipeline.splits import assign_groups, verify_group_isolation, detect_pair_leakage

# Register extra templates into TEMPLATES
import training.dataset_pipeline.extra_templates  # noqa: F401

from training.generate_dataset import (
    SEED,
    TEMPLATES,
    generate_off_by_one_mutations,
    generate_wrong_condition_mutations,
    generate_wrong_initialization_mutations,
    generate_wrong_operator_mutations,
    run_test_safely,
    validate_syntax,
    _generate_style_variants,
)


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def generate_classification_records(seed: int = SEED) -> tuple[List[dict], List[dict], List[dict]]:
    """Returns (accepted samples, rejected log, review_queue)."""
    random.seed(seed)
    accepted: List[dict] = []
    rejected: List[dict] = []
    review_queue: List[dict] = []
    norm_hashes: Dict[str, str] = {}

    mutation_generators = [
        generate_wrong_initialization_mutations,
        generate_off_by_one_mutations,
        generate_wrong_operator_mutations,
        generate_wrong_condition_mutations,
    ]

    sample_counter = 0

    for template in TEMPLATES:
        family_id = template["problem_type"]
        if not validate_syntax(template["code"]):
            rejected.append({"template_id": template["template_id"], "reason": "template_syntax"})
            continue
        passed, err = run_test_safely(template["code"], template["test_code"])
        if not passed:
            rejected.append({"template_id": template["template_id"], "reason": f"template_tests_failed: {err}"})
            continue

        sample_counter += 1
        accepted.append(
            _row(
                sample_counter,
                template,
                family_id,
                template["code"],
                "correct",
                "none",
                "Original correct implementation",
                "passes_all_tests",
                "validated",
            )
        )

        for gen_fn in mutation_generators:
            for mutation in gen_fn(template):
                mutated_code = mutation["code"]
                if not validate_syntax(mutated_code):
                    rejected.append(
                        {
                            "template_id": template["template_id"],
                            "mutation_type": mutation.get("mutation_type"),
                            "reason": "syntax_error",
                        }
                    )
                    continue

                passed_mut, err_mut = run_test_safely(mutated_code, template["test_code"])
                expected_pass = False
                if passed_mut:
                    review_queue.append(
                        {
                            "template_id": template["template_id"],
                            "intended_label": mutation["error_label"],
                            "reason": "mutation_still_passes_tests",
                            "mutation_type": mutation.get("mutation_type"),
                        }
                    )
                    continue

                nh = hashlib.sha256(normalize_code(mutated_code).encode()).hexdigest()
                if nh in norm_hashes:
                    rejected.append(
                        {
                            "template_id": template["template_id"],
                            "reason": f"normalized_duplicate_of_{norm_hashes[nh]}",
                        }
                    )
                    continue
                norm_hashes[nh] = f"S{sample_counter:04d}"

                sample_counter += 1
                accepted.append(
                    _row(
                        sample_counter,
                        template,
                        family_id,
                        mutated_code,
                        mutation["error_label"],
                        mutation.get("mutation_type", "unknown"),
                        mutation.get("mutation_description", ""),
                        "fails_tests",
                        "validated",
                    )
                )

    for variant in _generate_style_variants(TEMPLATES):
        family_id = variant["problem_type"]
        nh = hashlib.sha256(normalize_code(variant["code"]).encode()).hexdigest()
        if nh in norm_hashes:
            continue
        sample_counter += 1
        variant_row = _row(
            sample_counter,
            {"template_id": variant["source_template_id"], "problem_type": family_id},
            family_id,
            variant["code"],
            "correct",
            variant.get("mutation_type", "style_variant"),
            variant.get("mutation_description", ""),
            "passes_all_tests",
            "validated",
        )
        accepted.append(variant_row)

    families = [s["family_id"] for s in accepted]
    group_map = assign_groups(families, seed=seed)
    for s in accepted:
        s["split"] = group_map.get(s["family_id"], "train")
        s["dataset_version"] = DATASET_VERSION

    ok, iso_errors = verify_group_isolation(accepted, "family_id")
    if not ok:
        for e in iso_errors:
            rejected.append({"reason": f"split_isolation:{e}"})

    return accepted, rejected, review_queue


def _row(counter, template, family_id, code, label, mut_type, mut_desc, expected, status):
    return {
        "sample_id": f"S{counter:04d}",
        "source_template_id": template["template_id"],
        "family_id": family_id,
        "problem_type": template["problem_type"] if isinstance(template, dict) and "problem_type" in template else family_id,
        "code": code,
        "error_label": label,
        "mutation_type": mut_type,
        "mutation_description": mut_desc,
        "expected_behavior": expected,
        "validation_status": status,
    }


def write_outputs(
    root: Path,
    samples: List[dict],
    rejected: List[dict],
    review_queue: List[dict],
    pairs: List[dict],
    seed: int,
    dataset_version: str | None = None,
) -> Dict[str, Any]:
    version = dataset_version or DATASET_VERSION
    clf_dir = root / "classification"
    sim_dir = root / "similarity"
    clf_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_id",
        "source_template_id",
        "family_id",
        "problem_type",
        "code",
        "error_label",
        "mutation_type",
        "mutation_description",
        "expected_behavior",
        "validation_status",
        "split",
        "dataset_version",
    ]
    csv_path = clf_dir / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k, "") for k in fields})

    pair_fields = [
        "pair_id",
        "code_a",
        "code_b",
        "label",
        "pair_type",
        "family_a",
        "family_b",
        "split",
        "source",
        "dataset_version",
    ]
    pairs_path = sim_dir / "pairs.csv"
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        for p in pairs:
            row = {k: p.get(k, "") for k in pair_fields}
            row["dataset_version"] = version
            w.writerow(row)

    with open(clf_dir / "rejected.json", "w") as f:
        json.dump(rejected, f, indent=2)
    with open(clf_dir / "review_queue.json", "w") as f:
        json.dump(review_queue, f, indent=2)

    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    families = sorted(set(s["family_id"] for s in samples))
    split_families = defaultdict(set)
    for s in samples:
        split_families[s["split"]].add(s["family_id"])

    manifest = {
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "checksum_sha256": checksum,
        "grouping_key": "family_id (problem_type)",
        "independent_families": len(families),
        "families_per_split": {k: sorted(v) for k, v in split_families.items()},
        "note": "v1 test set was used in prior dev; this benchmark uses fresh family-level holdout.",
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def build_quality_report(
    samples: List[dict],
    rejected: List[dict],
    review_queue: List[dict],
    pairs: List[dict],
    manifest: dict,
) -> dict:
    class_dist = dict(Counter(s["error_label"] for s in samples))
    split_counts = dict(Counter(s["split"] for s in samples))
    cross = sum(1 for p in pairs if p.get("split") == "cross_split")
    eval_pairs = [p for p in pairs if p.get("split") in ("train", "val", "test")]

    sample_hashes = {code_hash(s["code"]): s["split"] for s in samples}
    pair_ok, pair_errors = detect_pair_leakage(
        [p for p in pairs if p.get("split") in ("train", "val", "test")],
        sample_hashes,
        code_hash,
    )

    pair_types = dict(Counter(p.get("pair_type") for p in pairs))
    label_dist = {"similar": sum(1 for p in eval_pairs if int(p["label"]) == 1), "dissimilar": sum(1 for p in eval_pairs if int(p["label"]) == 0)}

    return {
        "dataset_version": DATASET_VERSION,
        "total_accepted": len(samples),
        "total_rejected": len(rejected),
        "review_queue_count": len(review_queue),
        "human_reviewed": False,
        "class_distribution": class_dist,
        "split_sample_counts": split_counts,
        "independent_program_families": manifest.get("independent_families"),
        "families_per_split": manifest.get("families_per_split"),
        "syntax_validity_rate": 1.0,
        "behavioral_validation": "template unit tests in subprocess; finite tests — not universal proof",
        "duplicate_rejection_count": sum(1 for r in rejected if "duplicate" in r.get("reason", "")),
        "similarity_total_pairs": len(pairs),
        "similarity_cross_split_excluded": cross,
        "similarity_evaluable_pairs": len(eval_pairs),
        "similarity_evaluable_by_split": dict(Counter(p["split"] for p in eval_pairs)),
        "similarity_label_balance_evaluable": label_dist,
        "pair_type_counts": pair_types,
        "pair_leakage_check_passed": pair_ok,
        "pair_leakage_errors": pair_errors[:20],
        "seed": manifest.get("seed"),
        "checksum_sha256": manifest.get("checksum_sha256"),
        "limitations": [
            "Automated labels — not human-reviewed unless listed in review_queue.",
            "Behavioral tests are finite; passing does not prove full correctness.",
            "Similarity label 1 means same-family correct pairs or typed relation — not proven equivalence.",
        ],
    }


def main() -> int:
    root = PROJECT_ROOT / "data" / DATASET_VERSION
    print(f"Generating {DATASET_VERSION} under {root}")

    samples, rejected, review_queue = generate_classification_records(seed=SEED)
    pairs = generate_similarity_pairs_v2(samples, seed=SEED)
    manifest = write_outputs(root, samples, rejected, review_queue, pairs, SEED)

    report = build_quality_report(samples, rejected, review_queue, pairs, manifest)
    report_path = root / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in report if k != "pair_leakage_errors"}, indent=2))
    print(f"[OK] classification: {len(samples)} samples, {report['independent_program_families']} families")
    print(f"[OK] similarity: {report['similarity_total_pairs']} pairs, evaluable {report['similarity_evaluable_pairs']}")
    print(f"[OK] quality report: {report_path}")
    return 0 if report["pair_leakage_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
