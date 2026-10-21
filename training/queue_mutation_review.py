"""
Review workflow extension for generator mutations (including queue-only, no CSV row).

Decisions stored separately from sample-based review_decisions.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.generate_dataset import TEMPLATES, run_test_safely
from training.mutation_identity import code_hash_sha256, stable_mutation_id_from_parts
from training.phase13_queue_reconciliation import build_reconciliation_report
from training.review_workflow import VALID_STATUS, template_spec, validate_decision_payload

QUEUE_DECISIONS_PATH = PROJECT_ROOT / "data/benchmark_v3/queue_mutation_decisions.json"
REGISTRY_PATH = PROJECT_ROOT / "artifacts/metrics/phase14/mutation_registry.json"

TEMPLATE_BY_ID = {t["template_id"]: t for t in TEMPLATES}

# Evidence semantics (not equivalent conclusions)
EVIDENCE_SEMANTICS = {
    "bundled_tests_pass": "Finite generator tests completed with exit code 0.",
    "bundled_tests_fail": "Finite generator tests failed (assertion or runtime error).",
    "logically_correct_under_spec": "Requires human — tests passing does not prove logical correctness vs intended behavior.",
    "fault_label_matches_intended_taxonomy": "Requires human — label is a taxonomy claim, not test outcome.",
}


def _mutation_code_from_registry_record(rec: Dict[str, Any]) -> Optional[str]:
    """Re-resolve code from generator for registry records (read-only)."""
    from training.phase13_queue_reconciliation import MUTATION_GENERATORS

    tid = rec.get("template_id")
    template = TEMPLATE_BY_ID.get(tid)
    if not template:
        return None
    target_hash = rec.get("code_hash_sha256")
    for gen in MUTATION_GENERATORS:
        for mut in gen(template):
            ch = code_hash_sha256(mut["code"])
            if ch == target_hash:
                return mut["code"]
    return None


def build_mutation_registry() -> Dict[str, Any]:
    """One record per generator variant tied to the 10-item review queue."""
    recon = build_reconciliation_report()
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(recon.get("items") or [], start=1):
        q = item["queue_item"]
        queue_ref = f"review_queue[{idx - 1}]"
        mismatch = item.get("mismatch_class")
        for gm in item.get("generator_matches") or []:
            parts = {
                "template_id": q["template_id"],
                "error_label": gm["error_label"],
                "mutation_type": gm.get("mutation_type"),
                "code_hash_sha256": gm["code_hash_sha256"],
            }
            mid = stable_mutation_id_from_parts(parts)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            csv_links = [
                x
                for x in item.get("csv_code_hash_links") or []
                if x.get("code_hash_sha256") == gm["code_hash_sha256"]
            ]
            queue_flagged = (
                q.get("reason") == "mutation_still_passes_tests"
                and gm.get("bundled_tests_pass")
                and mismatch in ("queue_only_excluded_mutation", "taxonomy_collision_multiple_mutations")
            )
            records.append(
                {
                    "mutation_id": mid,
                    "queue_index": idx - 1,
                    "queue_ref": queue_ref,
                    "mismatch_class": mismatch,
                    "template_id": q["template_id"],
                    "intended_label": q["intended_label"],
                    "queue_mutation_type": q.get("mutation_type"),
                    "error_label": gm["error_label"],
                    "mutation_type": gm.get("mutation_type"),
                    "mutation_description": gm.get("mutation_description"),
                    "code_hash_sha256": gm["code_hash_sha256"],
                    "bundled_tests_pass": gm.get("bundled_tests_pass"),
                    "bundled_test_error": gm.get("bundled_test_error"),
                    "csv_sample_ids": [x["sample_id"] for x in csv_links],
                    "queue_reason_flagged": queue_flagged,
                    "in_dataset_csv": bool(csv_links),
                    "evidence_semantics_note": (
                        "bundled_tests_pass/fail is not equivalent to label correctness or logical incorrectness."
                    ),
                }
            )

    records.sort(key=lambda r: r["mutation_id"])
    return {
        "registry_version": "phase14.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_reviewed": False,
        "source_reconciliation": "artifacts/metrics/phase13/queue_reconciliation.json",
        "record_count": len(records),
        "records": records,
    }


def _default_queue_decisions(registry: Dict[str, Any]) -> Dict[str, Any]:
    decisions = []
    for i, rec in enumerate(registry.get("records") or [], start=1):
        decisions.append(
            {
                "review_id": f"QMRQ{i:03d}",
                "mutation_id": rec["mutation_id"],
                "status": "pending",
                "reviewer_id": None,
                "rationale": None,
                "decided_at": None,
                "anonymous_review": False,
                "audit_history": [],
                "provenance": {
                    "template_id": rec["template_id"],
                    "code_hash_sha256": rec["code_hash_sha256"],
                    "queue_ref": rec.get("queue_ref"),
                    "csv_sample_ids": rec.get("csv_sample_ids") or [],
                    "not_a_csv_sample_id": not rec.get("in_dataset_csv"),
                },
            }
        )
    return {
        "dataset_version": "benchmark_v3",
        "workflow": "queue_mutation_extension",
        "human_reviewed": False,
        "adjudication_complete": False,
        "note": "Decisions apply to generator mutations; does not modify dataset.csv.",
        "decisions": decisions,
    }


def load_queue_mutation_decisions(create_if_missing: bool = True) -> Dict[str, Any]:
    if QUEUE_DECISIONS_PATH.is_file():
        return json.loads(QUEUE_DECISIONS_PATH.read_text(encoding="utf-8"))
    reg = build_mutation_registry()
    data = _default_queue_decisions(reg)
    if create_if_missing:
        QUEUE_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_DECISIONS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def save_queue_mutation_decisions(data: Dict[str, Any]) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    QUEUE_DECISIONS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_registry_record(registry: Dict[str, Any], mutation_id: str) -> Optional[Dict[str, Any]]:
    for rec in registry.get("records") or []:
        if rec.get("mutation_id") == mutation_id:
            return rec
    return None


def build_queue_mutation_packet(
    decision_row: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    rec = find_registry_record(registry, decision_row["mutation_id"])
    if not rec:
        return {"error": "mutation_id not in registry", "mutation_id": decision_row.get("mutation_id")}

    tid = rec["template_id"]
    spec = template_spec(tid)
    code = _mutation_code_from_registry_record(rec)

    test_run = None
    if code and TEMPLATE_BY_ID.get(tid):
        passed, err = run_test_safely(code, TEMPLATE_BY_ID[tid]["test_code"])
        test_run = {"bundled_tests_pass": passed, "error_preview": (err or "")[:500] if err else None}

    conflicting = [
        r
        for r in registry.get("records") or []
        if r["template_id"] == tid
        and r["error_label"] == rec["error_label"]
        and r["mutation_type"] == rec["mutation_type"]
        and r["mutation_id"] != rec["mutation_id"]
    ]

    decision_needed = []
    if rec.get("queue_reason_flagged") and rec.get("bundled_tests_pass"):
        decision_needed.append(
            "Queue flagged mutation_still_passes_tests: accept for benchmark with stronger tests, "
            "reject as unsuitable, relabel, or needs_expert_review."
        )
    if conflicting:
        decision_needed.append(
            "Taxonomy collision: another variant shares label+mutation_type; confirm which variant queue entry refers to."
        )
    if rec.get("in_dataset_csv"):
        decision_needed.append(
            "CSV row exists for this code hash — confirm label and whether queue reason is stale."
        )
    if not decision_needed:
        decision_needed.append("Confirm fault taxonomy and suitability for benchmark export.")

    return {
        "review_id": decision_row.get("review_id"),
        "mutation_id": rec["mutation_id"],
        "batch": "A" if not rec.get("in_dataset_csv") else "B",
        "mismatch_class": rec.get("mismatch_class"),
        "status": decision_row.get("status", "pending"),
        "intended_behavior": spec,
        "intended_behavior_note": "Registry excerpt if present; family NL spec may be draft in catalog.",
        "candidate_code": code,
        "mutation_type": rec.get("mutation_type"),
        "assigned_error_label": rec.get("error_label"),
        "mutation_description": rec.get("mutation_description"),
        "provenance": {
            "template_id": tid,
            "code_hash_sha256": rec["code_hash_sha256"],
            "queue_ref": rec.get("queue_ref"),
            "csv_sample_ids": rec.get("csv_sample_ids"),
            "in_dataset_csv": rec.get("in_dataset_csv"),
        },
        "test_evidence": {
            "bundled_tests_pass": rec.get("bundled_tests_pass"),
            "bundled_test_error": rec.get("bundled_test_error"),
            "live_rerun": test_run,
            "semantics": EVIDENCE_SEMANTICS,
        },
        "conflicting_variants": [
            {
                "mutation_id": c["mutation_id"],
                "bundled_tests_pass": c.get("bundled_tests_pass"),
                "mutation_description": c.get("mutation_description"),
                "csv_sample_ids": c.get("csv_sample_ids"),
            }
            for c in conflicting
        ],
        "decision_needed": decision_needed,
        "decision_fields": {
            "allowed_statuses": sorted(VALID_STATUS - {"pending"}),
            "cli_example": (
                "py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue decide "
                f"--id {decision_row.get('review_id')} --status <status> --rationale \"...\" --reviewer <name>"
            ),
        },
        "audit_note": "Recording a decision appends audit_history; dataset.csv unchanged.",
    }


def record_queue_decision(
    review_id: str,
    status: str,
    rationale: str,
    reviewer: Optional[str],
    anonymous: bool,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    rev = reviewer if not anonymous else "anonymous"
    ok, msg = validate_decision_payload(status, rationale, None, rev, anonymous)
    if not ok:
        return False, msg

    data = load_queue_mutation_decisions(create_if_missing=True)
    for d in data["decisions"]:
        if d["review_id"] != review_id:
            continue
        if d["status"] != "pending" and not force:
            return False, "Already decided; use --force to append audit_history"
        prior = {
            "status": d.get("status"),
            "rationale": d.get("rationale"),
            "reviewer_id": d.get("reviewer_id"),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        d.setdefault("audit_history", []).append(prior)
        d["status"] = status
        d["rationale"] = rationale
        d["reviewer_id"] = rev
        d["decided_at"] = datetime.now(timezone.utc).isoformat()
        d["anonymous_review"] = anonymous
        data["human_reviewed"] = any(x["status"] not in ("pending", None) for x in data["decisions"])
        data["adjudication_complete"] = all(x["status"] != "pending" for x in data["decisions"])
        save_queue_mutation_decisions(data)
        return True, "ok"
    return False, f"Unknown review id {review_id}"


def write_registry_artifact() -> Path:
    reg = build_mutation_registry()
    out = PROJECT_ROOT / "artifacts/metrics/phase14"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mutation_registry.json"
    path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return path
