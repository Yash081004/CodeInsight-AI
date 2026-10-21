"""
Human label review workflow (decisions are never auto-filled).

Usage:
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 list
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --id V3RQ001
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 verify
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --reviewer alice

Queue mutation extension (no CSV sample_id):
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue list
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue packet --id QMRQ001
    py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue decide --id QMRQ001 --status rejected --rationale "..." --reviewer alice
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.review_workflow import (
    VALID_STATUS,
    build_review_packet,
    decisions_path,
    load_decisions,
    validate_decision_payload,
    verify_workflow,
)

from backend.app.config import ERROR_LABELS


def _paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "benchmark_v3":
        return (
            PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json",
            PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "review_queue.json",
        )
    return (
        PROJECT_ROOT / "data" / "benchmark_v2" / "review_decisions.json",
        PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "review_queue.json",
    )


def _load_decisions(dataset: str = "benchmark_v2") -> dict:
    decisions_path_file, review_queue_path = _paths(dataset)
    if decisions_path_file.exists():
        return json.loads(decisions_path_file.read_text(encoding="utf-8"))
    items = []
    if review_queue_path.exists():
        rq = json.loads(review_queue_path.read_text(encoding="utf-8"))
        for i, row in enumerate(rq, start=1):
            items.append(
                {
                    "review_id": f"RQ{i:03d}",
                    "template_id": row.get("template_id"),
                    "intended_label": row.get("intended_label"),
                    "mutation_type": row.get("mutation_type"),
                    "queue_reason": row.get("reason"),
                    "status": "pending",
                    "reviewer_id": None,
                    "rationale": None,
                    "original_code_ref": f"template:{row.get('template_id')}",
                    "mutated_code_ref": "not materialized in queue — regenerate from template + mutation_type",
                    "behavioral_tests": "finite template tests; passing does not prove universal incorrectness",
                }
            )
    return {
        "dataset_version": dataset,
        "human_reviewed": False,
        "decisions": items,
        "note": "Accepted items require a new dataset version export; this file does not modify dataset.csv.",
    }


def _save(data: dict, dataset: str) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = decisions_path(dataset) if dataset == "benchmark_v3" else _paths(dataset)[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cmd_list(args) -> int:
    data = _load_decisions(args.dataset)
    dp, _ = _paths(args.dataset)
    if not dp.exists():
        _save(data, args.dataset)
    for d in data["decisions"]:
        ref = d.get("sample_id") or d.get("template_id")
        lbl = d.get("current_label") or d.get("intended_label")
        print(f"{d['review_id']}\t{d['status']}\t{ref}\t{lbl}")
    return 0


def cmd_show(args) -> int:
    data = _load_decisions(args.dataset)
    for d in data["decisions"]:
        if d["review_id"] == args.id:
            print(json.dumps(d, indent=2))
            return 0
    print(f"Unknown id {args.id}")
    return 1


def cmd_packet(args) -> int:
    data = _load_decisions(args.dataset)
    target = None
    for d in data["decisions"]:
        if args.id and d["review_id"] == args.id:
            target = d
            break
        if args.sample and d.get("sample_id") == args.sample:
            target = d
            break
    if not target:
        print("Unknown review id or sample")
        return 1
    pkt = build_review_packet(target, include_model_hints=args.show_model_hints)
    print(json.dumps(pkt, indent=2))
    return 0


def cmd_verify(_args) -> int:
    print(json.dumps(verify_workflow(), indent=2))
    return 0


def cmd_queue_list(args) -> int:
    if args.dataset != "benchmark_v3":
        print("Queue mutation review is benchmark_v3 only.")
        return 1
    from training.queue_mutation_review import load_queue_mutation_decisions, write_registry_artifact

    write_registry_artifact()
    data = load_queue_mutation_decisions(create_if_missing=True)
    for d in data["decisions"]:
        print(f"{d['review_id']}\t{d['status']}\t{d['mutation_id']}")
    return 0


def cmd_queue_packet(args) -> int:
    if args.dataset != "benchmark_v3":
        print("Queue mutation review is benchmark_v3 only.")
        return 1
    from training.queue_mutation_review import (
        build_mutation_registry,
        build_queue_mutation_packet,
        load_queue_mutation_decisions,
    )

    reg = build_mutation_registry()
    data = load_queue_mutation_decisions(create_if_missing=True)
    target = None
    for d in data["decisions"]:
        if d["review_id"] == args.id:
            target = d
            break
    if not target:
        print(f"Unknown id {args.id}")
        return 1
    print(json.dumps(build_queue_mutation_packet(target, reg), indent=2))
    return 0


def cmd_queue_decide(args) -> int:
    if args.dataset != "benchmark_v3":
        print("Queue mutation review is benchmark_v3 only.")
        return 1
    from training.queue_mutation_review import record_queue_decision

    ok, msg = record_queue_decision(
        args.id,
        args.status,
        args.rationale,
        args.reviewer,
        args.anonymous_review,
        force=args.force,
    )
    if not ok:
        print(f"[FAIL] {msg}")
        return 1
    print(f"[OK] {args.id} -> {args.status}")
    return 0


def cmd_decide(args) -> int:
    anonymous = args.anonymous_review
    reviewer = args.reviewer if not anonymous else "anonymous"
    ok, msg = validate_decision_payload(
        args.status,
        args.rationale,
        args.new_label,
        reviewer,
        anonymous,
    )
    if not ok:
        print(f"[FAIL] {msg}")
        return 1
    if args.new_label and args.new_label not in ERROR_LABELS:
        print(f"Invalid label; use one of {ERROR_LABELS}")
        return 1

    data = _load_decisions(args.dataset)
    for d in data["decisions"]:
        if d["review_id"] == args.id:
            if d["status"] != "pending" and not args.force:
                print("Already decided; use --force to append new decision via audit_history")
                return 1
            prior = {
                "status": d.get("status"),
                "rationale": d.get("rationale"),
                "reviewer_id": d.get("reviewer_id"),
                "proposed_label": d.get("proposed_label"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            d.setdefault("audit_history", []).append(prior)
            d["status"] = args.status
            d["rationale"] = args.rationale
            d["reviewer_id"] = reviewer
            d["decided_at"] = datetime.now(timezone.utc).isoformat()
            d["anonymous_review"] = anonymous
            if args.new_label:
                d["proposed_label"] = args.new_label
            if args.status == "accepted":
                d["export_note"] = "Export via build_reviewed_eval_slice; dataset.csv unchanged."
            data["human_reviewed"] = any(x["status"] not in ("pending", None) for x in data["decisions"])
            data["adjudication_complete"] = all(x["status"] != "pending" for x in data["decisions"])
            _save(data, args.dataset)
            print(f"[OK] {args.id} -> {args.status}")
            return 0
    print(f"Unknown id {args.id}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark_v2", choices=["benchmark_v2", "benchmark_v3"])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("verify")
    p_show = sub.add_parser("show")
    p_show.add_argument("--id", required=True)
    p_pkt = sub.add_parser("packet")
    p_pkt.add_argument("--id", default=None)
    p_pkt.add_argument("--sample", default=None)
    p_pkt.add_argument("--show-model-hints", action="store_true")
    p_dec = sub.add_parser("decide")
    p_dec.add_argument("--id", required=True)
    p_dec.add_argument("--status", required=True)
    p_dec.add_argument("--rationale", default="")
    p_dec.add_argument("--reviewer", default=None)
    p_dec.add_argument("--anonymous-review", action="store_true")
    p_dec.add_argument("--new-label", default=None, dest="new_label")
    p_dec.add_argument("--force", action="store_true")
    p_queue = sub.add_parser("queue", help="Review queue-only generator mutations (QMRQ ids)")
    queue_sub = p_queue.add_subparsers(dest="queue_cmd", required=True)
    queue_sub.add_parser("list")
    p_q_pkt = queue_sub.add_parser("packet")
    p_q_pkt.add_argument("--id", required=True)
    p_q_dec = queue_sub.add_parser("decide")
    p_q_dec.add_argument("--id", required=True)
    p_q_dec.add_argument("--status", required=True)
    p_q_dec.add_argument("--rationale", default="")
    p_q_dec.add_argument("--reviewer", default=None)
    p_q_dec.add_argument("--anonymous-review", action="store_true")
    p_q_dec.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cmd == "queue":
        if args.queue_cmd == "list":
            return cmd_queue_list(args)
        if args.queue_cmd == "packet":
            return cmd_queue_packet(args)
        if args.queue_cmd == "decide":
            return cmd_queue_decide(args)
        return 1
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "packet":
        if not args.id and not args.sample:
            print("Provide --id or --sample")
            return 1
        return cmd_packet(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "decide":
        return cmd_decide(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
