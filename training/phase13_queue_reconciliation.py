"""
Phase 13: trace review_queue entries to generator mutations vs CSV rows (read-only).

Queue items with reason mutation_still_passes_tests refer to mutations excluded from
dataset.csv at generation time — not missing CSV rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.generate_dataset import (
    TEMPLATES,
    generate_off_by_one_mutations,
    generate_wrong_condition_mutations,
    generate_wrong_initialization_mutations,
    generate_wrong_operator_mutations,
    run_test_safely,
)
from training.phase12_template_tests import run_template_tests_on_code

QUEUE_PATH = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
V3_CSV = PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv"

MUTATION_GENERATORS = (
    generate_wrong_initialization_mutations,
    generate_off_by_one_mutations,
    generate_wrong_operator_mutations,
    generate_wrong_condition_mutations,
)

TEMPLATE_BY_ID = {t["template_id"]: t for t in TEMPLATES}


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


def enumerate_mutations(template_id: str) -> List[Dict[str, Any]]:
    template = TEMPLATE_BY_ID.get(template_id)
    if not template:
        return []
    out: List[Dict[str, Any]] = []
    for gen in MUTATION_GENERATORS:
        for mut in gen(template):
            passed, err = run_test_safely(mut["code"], template["test_code"])
            out.append(
                {
                    "error_label": mut["error_label"],
                    "mutation_type": mut.get("mutation_type"),
                    "mutation_description": mut.get("mutation_description", ""),
                    "code_hash_sha256": code_hash(mut["code"]),
                    "bundled_tests_pass": bool(passed),
                    "bundled_test_error": err if not passed else None,
                }
            )
    return out


def _csv_rows_for_queue_key(
    df: pd.DataFrame, template_id: str, intended_label: str, mutation_type: str
) -> pd.DataFrame:
    m = (df["source_template_id"] == template_id) & (df["error_label"] == intended_label)
    if mutation_type and "mutation_type" in df.columns:
        m = m & (df["mutation_type"] == mutation_type)
    return df[m]


def _match_generator_mutations(
    mutations: List[Dict[str, Any]], intended_label: str, mutation_type: str
) -> List[Dict[str, Any]]:
    return [
        m
        for m in mutations
        if m["error_label"] == intended_label and m.get("mutation_type") == mutation_type
    ]


def reconcile_queue_item(df: pd.DataFrame, item: Dict[str, Any]) -> Dict[str, Any]:
    tid = item["template_id"]
    label = item["intended_label"]
    mut_type = item.get("mutation_type", "")
    reason = item.get("reason", "")

    csv_strict = _csv_rows_for_queue_key(df, tid, label, mut_type)
    all_template_rows = df[df["source_template_id"] == tid]
    mutations = enumerate_mutations(tid)
    gen_matches = _match_generator_mutations(mutations, label, mut_type)

    gen_pass = [m for m in gen_matches if m["bundled_tests_pass"]]
    gen_fail = [m for m in gen_matches if not m["bundled_tests_pass"]]

    csv_hash_matches: List[Dict[str, Any]] = []
    for _, row in csv_strict.iterrows():
        rh = code_hash(str(row["code"]))
        for gm in gen_matches:
            if gm["code_hash_sha256"] == rh:
                csv_hash_matches.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "code_hash_sha256": rh,
                        "generator_tests_pass": gm["bundled_tests_pass"],
                        "mutation_description": gm["mutation_description"],
                    }
                )

    mismatch_class: str
    evidence: List[str] = []

    if not TEMPLATE_BY_ID.get(tid):
        mismatch_class = "unknown_template_id"
        evidence.append(f"No template {tid} in generator registry.")
    elif not gen_matches:
        mismatch_class = "mutation_registry_drift"
        evidence.append(
            "No generator mutation matches queue (template_id, intended_label, mutation_type)."
        )
    elif len(csv_strict) == 0 and gen_pass and reason == "mutation_still_passes_tests":
        mismatch_class = "queue_only_excluded_mutation"
        evidence.append(
            "Generator produced mutation(s) that still pass bundled tests; "
            "generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead."
        )
        evidence.append(
            f"Matched {len(gen_pass)} passing mutation variant(s); 0 CSV rows with same taxonomy key."
        )
    elif len(csv_strict) > 0 and gen_pass and gen_fail:
        mismatch_class = "taxonomy_collision_multiple_mutations"
        evidence.append(
            "Multiple distinct mutations share the same error_label + mutation_type; "
            "CSV may contain one variant while queue flags another."
        )
        if csv_hash_matches:
            evidence.append(
                f"CSV row hash-linked to generator: {csv_hash_matches[0]['sample_id']} "
                f"(tests_pass={csv_hash_matches[0]['generator_tests_pass']})."
            )
        elif len(gen_pass):
            evidence.append("Queue likely refers to a passing variant not present in CSV.")
    elif len(csv_strict) > 0 and csv_hash_matches:
        linked = csv_hash_matches[0]
        if reason == "mutation_still_passes_tests" and not linked["generator_tests_pass"]:
            mismatch_class = "stale_queue_reason_vs_csv_sample"
            evidence.append(
                "Queue reason mutation_still_passes_tests but hash-linked CSV sample fails bundled tests today."
            )
            evidence.append(
                f"Linked sample {linked['sample_id']} ({linked.get('split')}): {linked.get('mutation_description')}"
            )
        else:
            mismatch_class = "csv_row_matches_taxonomy_key"
            evidence.append("CSV contains row(s) matching queue taxonomy key and generator code hash.")
    elif len(csv_strict) > 0:
        mismatch_class = "csv_row_matches_taxonomy_key"
        evidence.append("CSV contains row(s) matching queue taxonomy key (no generator hash link).")
    else:
        mismatch_class = "needs_manual_trace"
        evidence.append("Partial match; inspect generator_matches and csv_strict.")

    recommended_next_action = "requires_reviewer_intended_behavior"
    if mismatch_class == "queue_only_excluded_mutation":
        recommended_next_action = "requires_reviewer_intended_behavior"
    elif mismatch_class == "mutation_registry_drift":
        recommended_next_action = "insufficient_evidence"

    return {
        "queue_item": item,
        "mismatch_class": mismatch_class,
        "evidence": evidence,
        "csv_strict_match_count": int(len(csv_strict)),
        "csv_strict_sample_ids": csv_strict["sample_id"].tolist() if len(csv_strict) else [],
        "csv_template_row_count": int(len(all_template_rows)),
        "csv_template_labels": sorted(
            all_template_rows[["error_label", "mutation_type"]].drop_duplicates().astype(str).agg(":".join, axis=1).tolist()
        ),
        "generator_match_count": len(gen_matches),
        "generator_passing_variant_count": len(gen_pass),
        "generator_failing_variant_count": len(gen_fail),
        "generator_matches": gen_matches,
        "csv_code_hash_links": csv_hash_matches,
        "recommended_next_action": recommended_next_action,
        "human_reviewed": False,
    }


def build_reconciliation_report() -> Dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(V3_CSV)
    rows = [reconcile_queue_item(df, item) for item in queue]
    by_class: Dict[str, int] = {}
    for r in rows:
        by_class[r["mismatch_class"]] = by_class.get(r["mismatch_class"], 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_reviewed": False,
        "source_queue_path": str(QUEUE_PATH.relative_to(PROJECT_ROOT)),
        "source_csv_path": str(V3_CSV.relative_to(PROJECT_ROOT)),
        "summary_by_mismatch_class": by_class,
        "items": rows,
    }


def main() -> int:
    report = build_reconciliation_report()
    out = PROJECT_ROOT / "artifacts/metrics/phase13"
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'queue_reconciliation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
