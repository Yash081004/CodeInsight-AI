"""
Load Phase 13 authoring packets and merge draft catalog entries (not human-reviewed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKETS_DIR = PROJECT_ROOT / "data" / "specifications" / "packets"
CATALOG_PATH = PROJECT_ROOT / "data" / "specifications" / "catalog.v1.json"


def load_authoring_packets() -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    if not PACKETS_DIR.is_dir():
        return packets
    for path in sorted(PACKETS_DIR.glob("*.authoring_packet.json")):
        packets.append(json.loads(path.read_text(encoding="utf-8")))
    return packets


def packet_to_catalog_entry(packet: Dict[str, Any]) -> Dict[str, Any]:
    fam = packet["family_id"]
    prov = dict(packet.get("provenance") or {})
    prov.setdefault("review_status", "draft")
    prov.setdefault("source", "machine_suggested")
    return {
        "spec_id": f"spec.{fam}.phase13_draft",
        "family_id": fam,
        "source_template_ids": packet.get("source_template_ids") or [],
        "problem_statement": packet["problem_statement"],
        "input_assumptions": packet.get("input_assumptions") or [],
        "expected_output_summary": packet.get("expected_output_summary") or "",
        "boundary_cases": packet.get("boundary_cases") or [],
        "provenance": prov,
        "associated_sample_ids": packet.get("associated_sample_ids") or [],
    }


def merge_packets_into_catalog(write: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    from training.spec_catalog import validate_catalog

    packets = load_authoring_packets()
    if CATALOG_PATH.is_file():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    else:
        catalog = {
            "spec_catalog_version": "v1.0",
            "generated_at": None,
            "human_reviewed": False,
            "entries": [],
        }

    by_fam = {e.get("family_id"): e for e in catalog.get("entries") or []}
    for packet in packets:
        entry = packet_to_catalog_entry(packet)
        fam = entry["family_id"]
        existing = by_fam.get(fam)
        if existing and (existing.get("provenance") or {}).get("review_status") == "human_reviewed":
            continue
        by_fam[fam] = entry

    catalog["entries"] = sorted(by_fam.values(), key=lambda e: e.get("family_id", ""))
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    catalog["human_reviewed"] = False
    catalog["note"] = (
        "Includes Phase 13 authoring packets (draft / machine_suggested). "
        "Not verified by human reviewers."
    )

    ok, errs = validate_catalog(catalog)
    if write and ok:
        CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog, errs


def main() -> int:
    catalog, errs = merge_packets_into_catalog(write=True)
    if errs:
        print("Catalog validation errors:", errs)
        return 1
    print(f"Catalog entries: {len(catalog.get('entries') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
