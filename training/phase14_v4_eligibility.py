"""
Benchmark v4 eligibility — independent dimensions (strict vs documented narrow protocol).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.benchmark_v4_build import V3_CSV, V3_QUEUE, _hash
from training.dataset_validation import sha256_file
from training.phase10_v3_exclusions import load_review_queue_template_ids
from training.phase12_template_tests import run_template_tests_on_code
from training.review_workflow import load_decisions
from training.spec_catalog import load_catalog

from training.benchmark_v4_build import human_reviewed_families


def _pending_sample_ids() -> Set[str]:
    pending: Set[str] = set()
    for d in load_decisions("benchmark_v3").get("decisions") or []:
        if d.get("status") == "pending" and d.get("sample_id"):
            pending.add(str(d["sample_id"]))
    return pending


def assess_row(row: Dict[str, Any], *, queue_tids: Set[str], reviewed_fams: Set[str], pending_samples: Set[str]) -> Dict[str, Any]:
    sid = str(row.get("sample_id", ""))
    tid = str(row.get("source_template_id", ""))
    fam = str(row.get("family_id", ""))
    label = str(row.get("error_label", ""))
    mut = str(row.get("mutation_type", ""))
    exp = str(row.get("expected_behavior", ""))

    spec_ok = fam in reviewed_fams
    not_queue_template = tid not in queue_tids
    not_style = mut != "style_variant"
    fault_row = label != "correct"
    fails_tests_field = exp == "fails_tests"
    correct_row = label == "correct"

    tr = run_template_tests_on_code(tid, str(row.get("code", ""))) if fault_row or correct_row else {}
    executed = tr.get("executed", False)
    tests_pass = tr.get("passed_all") if executed else None
    tests_fail = executed and not tr.get("passed_all")

    label_pending = sid in pending_samples

    dimensions = {
        "spec_eligibility": {
            "pass": spec_ok,
            "reason": None if spec_ok else "family_spec_not_human_reviewed",
        },
        "label_provenance_eligibility": {
            "pass": not label_pending,
            "reason": None if not label_pending else "sample_label_review_pending",
        },
        "mutation_validity_eligibility": {
            "pass": not_queue_template and not_style,
            "reasons": [
                r
                for r, cond in (
                    ("review_queue_template", not not_queue_template),
                    ("synthetic_style_variant", not not_style),
                )
                if cond
            ],
        },
        "test_evidence_alignment": {
            "pass": (tests_fail if fails_tests_field else tests_pass) if executed else None,
            "bundled_tests_pass": tests_pass,
            "expected_behavior_field": exp,
            "note": "Alignment is test outcome vs CSV expected_behavior — not label correctness.",
        },
    }

    # Strict v4 (Phase 13 rules): all dimensions + correct-row policy
    strict_blockers: List[str] = []
    if not spec_ok:
        strict_blockers.append("family_spec_not_human_reviewed")
    if not not_queue_template:
        strict_blockers.append("review_queue_template")
    if not not_style:
        strict_blockers.append("synthetic_style_variant")
    if correct_row:
        strict_blockers.append("correct_baseline_requires_spec_context")
    if label_pending:
        strict_blockers.append("sample_label_review_pending")

    strict_eligible = len(strict_blockers) == 0

    # Narrow documented protocol (NOT promoted as ready): fault rows with failing tests, no queue template
    narrow_blockers: List[str] = []
    if not fault_row:
        narrow_blockers.append("not_a_fault_row")
    if not fails_tests_field:
        narrow_blockers.append("expected_behavior_not_fails_tests")
    if not not_queue_template:
        narrow_blockers.append("review_queue_template")
    if not_style:
        narrow_blockers.append("synthetic_style_variant")
    if executed and tests_pass:
        narrow_blockers.append("bundled_tests_pass_on_fault_row")
    if label_pending:
        narrow_blockers.append("sample_label_review_pending")
    # Narrow still flags missing spec separately — does not auto-clear
    if not spec_ok:
        narrow_blockers.append("family_spec_not_human_reviewed")

    narrow_hypothetical = len([b for b in narrow_blockers if b != "family_spec_not_human_reviewed"]) == 0

    return {
        "sample_id": sid,
        "family_id": fam,
        "source_template_id": tid,
        "error_label": label,
        "dimensions": dimensions,
        "strict_v4_eligible": strict_eligible,
        "strict_v4_blockers": strict_blockers,
        "narrow_protocol_blockers": narrow_blockers,
        "narrow_protocol_eligible_except_spec": narrow_hypothetical and not spec_ok,
        "approved_spec_does_not_clear_label_concern": label_pending or tid in queue_tids,
    }


def build_eligibility_report(*, write_csv: bool = True) -> Dict[str, Any]:
    catalog = load_catalog()
    reviewed = human_reviewed_families(catalog)
    queue_tids = load_review_queue_template_ids(V3_QUEUE)
    pending = _pending_sample_ids()
    df = pd.read_csv(V3_CSV)

    per_row = [assess_row(r.to_dict(), queue_tids=queue_tids, reviewed_fams=reviewed, pending_samples=pending) for _, r in df.iterrows()]

    reason_counter: Counter = Counter()
    for pr in per_row:
        for b in pr.get("strict_v4_blockers") or []:
            reason_counter[b] += 1

    independent_reason_counts: Counter = Counter()
    for pr in per_row:
        reasons_for_row = set(pr.get("strict_v4_blockers") or [])
        if not pr["dimensions"]["spec_eligibility"]["pass"]:
            reasons_for_row.add("family_spec_not_human_reviewed")
        for r in pr["dimensions"]["mutation_validity_eligibility"].get("reasons") or []:
            reasons_for_row.add(r)
        if not pr["dimensions"]["label_provenance_eligibility"]["pass"]:
            reasons_for_row.add("sample_label_review_pending")
        for r in reasons_for_row:
            independent_reason_counts[r] += 1
    independent_reason_counts = dict(independent_reason_counts)

    strict_count = sum(1 for pr in per_row if pr["strict_v4_eligible"])
    narrow_except_spec = sum(1 for pr in per_row if pr.get("narrow_protocol_eligible_except_spec"))

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_reviewed": False,
        "source_v3_checksum_sha256": sha256_file(V3_CSV),
        "strict_v4_eligible_count": strict_count,
        "narrow_protocol_eligible_except_spec_count": narrow_except_spec,
        "v4_ready_for_training": strict_count > 0,
        "pristine_test_claim": False,
        "independent_quarantine_reason_counts": independent_reason_counts,
        "strict_blocker_counts": dict(reason_counter),
        "protocol_notes": {
            "strict": "All blockers must be clear; includes human-reviewed family spec and correct-row policy.",
            "narrow": "Hypothetical fault-only slice; still requires human spec for spec+code training — not relaxed for row count.",
            "spec_vs_label": "human_reviewed spec does not clear pending sample review or review_queue_template concerns.",
        },
        "per_row": per_row,
    }

    out = PROJECT_ROOT / "artifacts/metrics/phase14"
    out.mkdir(parents=True, exist_ok=True)
    (out / "v4_eligibility_assessment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if write_csv:
        summary_rows = []
        for pr in per_row:
            summary_rows.append(
                {
                    "sample_id": pr["sample_id"],
                    "strict_v4_eligible": pr["strict_v4_eligible"],
                    "strict_v4_blockers": ";".join(pr["strict_v4_blockers"]),
                    "narrow_protocol_eligible_except_spec": pr.get("narrow_protocol_eligible_except_spec"),
                    "approved_spec_does_not_clear_label_concern": pr["approved_spec_does_not_clear_label_concern"],
                }
            )
        pd.DataFrame(summary_rows).to_csv(out / "v4_eligibility_by_sample.csv", index=False)

    return report


def main() -> int:
    r = build_eligibility_report()
    print(json.dumps({"strict_eligible": r["strict_v4_eligible_count"], "ready": r["v4_ready_for_training"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
