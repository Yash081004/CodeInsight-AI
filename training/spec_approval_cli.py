"""
Specification approval workflow (draft → human_reviewed) — never auto-approves.

Usage:
    py -3.13 -m training.spec_approval_cli list
    py -3.13 -m training.spec_approval_cli show --family factorial
    py -3.13 -m training.spec_approval_cli checklist --family factorial
    py -3.13 -m training.spec_approval_cli approve --family factorial --reviewer NAME --notes "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.phase13_spec_packets import PACKETS_DIR, load_authoring_packets
from training.spec_catalog import CATALOG_PATH, load_catalog, validate_catalog

CHECKLIST_FIELDS = [
    ("inputs_and_assumptions", "Are inputs and assumptions explicit and consistent across templates?"),
    ("intended_output", "Is intended output/behavior stated without referencing faulty code?"),
    ("boundary_cases", "Are boundary cases listed; are any still ambiguous?"),
    ("deterministic_testable", "Is behavior deterministic and objectively testable?"),
    ("provenance", "Will reviewer identity and timestamp be recorded on approval?"),
    ("version_history", "Is spec_id/version stable for catalog history?"),
]


def _find_catalog_entry(catalog: Dict, family_id: str) -> Dict | None:
    for e in catalog.get("entries") or []:
        if e.get("family_id") == family_id:
            return e
    return None


def _find_packet(family_id: str) -> Dict | None:
    for p in load_authoring_packets():
        if p.get("family_id") == family_id:
            return p
    return None


def cmd_list(_args) -> int:
    catalog = load_catalog()
    for e in catalog.get("entries") or []:
        rs = (e.get("provenance") or {}).get("review_status")
        print(f"{e.get('family_id')}\t{rs}\t{e.get('spec_id')}")
    return 0


def cmd_show(args) -> int:
    packet = _find_packet(args.family)
    entry = _find_catalog_entry(load_catalog(), args.family)
    print(json.dumps({"authoring_packet": packet, "catalog_entry": entry}, indent=2))
    return 0


def cmd_checklist(args) -> int:
    packet = _find_packet(args.family)
    entry = _find_catalog_entry(load_catalog(), args.family)
    ambiguities: List[str] = []
    if not packet:
        ambiguities.append("No authoring packet file found.")
    else:
        if "negative n" in (packet.get("problem_statement") or "").lower():
            ambiguities.append("Negative input handling may need explicit policy per template.")
        if packet.get("family_id") == "linear_search" and "Empty list" in str(packet.get("boundary_cases")):
            ambiguities.append("Empty-list target behavior should be confirmed against each template's tests.")
        if packet.get("family_id") == "gcd":
            ambiguities.append("Zero and sign handling must be aligned with gcd_1 vs gcd_iter tests.")

    rs = (entry or {}).get("provenance", {}).get("review_status") if entry else "missing"
    body = {
        "family_id": args.family,
        "current_review_status": rs,
        "human_reviewed": False,
        "checklist": [{ "field": f, "question": q } for f, q in CHECKLIST_FIELDS],
        "recorded_ambiguities": ambiguities,
        "approval_command": (
            f"py -3.13 -m training.spec_approval_cli approve --family {args.family} "
            "--reviewer <NAME> --notes \"...\""
        ),
        "note": "Do not approve until ambiguities are resolved or explicitly accepted by reviewer.",
    }
    print(json.dumps(body, indent=2))
    return 0


def cmd_approve(args) -> int:
    if not args.reviewer or not args.reviewer.strip():
        print("[FAIL] --reviewer is required for approve (no auto-approval).")
        return 1
    catalog = load_catalog()
    entry = _find_catalog_entry(catalog, args.family)
    if not entry:
        print(f"No catalog entry for family {args.family}")
        return 1
    if entry.get("provenance", {}).get("review_status") == "human_reviewed":
        print("Already human_reviewed; edit catalog manually with audit if revision needed.")
        return 1

    prov = dict(entry.get("provenance") or {})
    history = list(prov.get("status_history") or [])
    history.append(
        {
            "from": prov.get("review_status"),
            "to": "human_reviewed",
            "at": datetime.now(timezone.utc).isoformat(),
            "reviewer_id": args.reviewer,
            "notes": args.notes or "",
        }
    )
    prov["review_status"] = "human_reviewed"
    prov["reviewer_id"] = args.reviewer
    prov["source"] = "human_reviewer"
    prov["author"] = args.reviewer
    prov["status_history"] = history
    if args.notes:
        prov["notes"] = args.notes
    entry["provenance"] = prov

    catalog["human_reviewed"] = all(
        (e.get("provenance") or {}).get("review_status") == "human_reviewed"
        for e in catalog.get("entries") or []
    )
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()

    ok, errs = validate_catalog(catalog)
    if not ok:
        print("[FAIL] validation:", errs)
        return 1
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"[OK] {args.family} -> human_reviewed (reviewer={args.reviewer})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_show = sub.add_parser("show")
    p_show.add_argument("--family", required=True)
    p_chk = sub.add_parser("checklist")
    p_chk.add_argument("--family", required=True)
    p_app = sub.add_parser("approve")
    p_app.add_argument("--family", required=True)
    p_app.add_argument("--reviewer", required=True)
    p_app.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "checklist":
        return cmd_checklist(args)
    if args.cmd == "approve":
        return cmd_approve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
