"""
Problem specification catalog (v1) — backward compatible with legacy CSV rows lacking specs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "specifications" / "catalog.v1.json"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "problem_specification.v1.json"

VALID_REVIEW_STATUS = frozenset({"missing", "draft", "machine_suggested", "human_reviewed"})


def load_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or CATALOG_PATH
    if not p.is_file():
        return {
            "spec_catalog_version": "v1.0",
            "generated_at": None,
            "human_reviewed": False,
            "note": "No catalog file yet; legacy datasets use CSV fields only.",
            "entries": [],
        }
    return json.loads(p.read_text(encoding="utf-8"))


def validate_catalog(catalog: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if catalog.get("spec_catalog_version", "").startswith("v1") is False:
        errors.append("spec_catalog_version must start with v1.")
    if catalog.get("human_reviewed") is True:
        for e in catalog.get("entries") or []:
            prov = e.get("provenance") or {}
            if prov.get("review_status") != "human_reviewed":
                errors.append(f"{e.get('spec_id')}: human_reviewed catalog requires human_reviewed entries")
    for e in catalog.get("entries") or []:
        if not e.get("spec_id"):
            errors.append("entry missing spec_id")
        if not (e.get("problem_statement") or "").strip():
            errors.append(f"{e.get('spec_id')}: empty problem_statement")
        rs = (e.get("provenance") or {}).get("review_status")
        if rs not in VALID_REVIEW_STATUS - {"missing"}:
            if rs not in VALID_REVIEW_STATUS:
                errors.append(f"{e.get('spec_id')}: invalid review_status {rs}")
    return len(errors) == 0, errors


def spec_for_family(catalog: Dict[str, Any], family_id: str) -> Optional[Dict[str, Any]]:
    for e in catalog.get("entries") or []:
        if e.get("family_id") == family_id:
            return e
    return None


def legacy_row_has_spec(row: Dict[str, Any], catalog: Optional[Dict[str, Any]] = None) -> bool:
    """Legacy CSV: expected_behavior is not a spec; check catalog by family."""
    cat = catalog or load_catalog()
    fam = row.get("family_id") or row.get("problem_type")
    ent = spec_for_family(cat, fam) if fam else None
    if not ent:
        return False
    return (ent.get("provenance") or {}).get("review_status") == "human_reviewed"


def build_initial_draft_catalog() -> Dict[str, Any]:
    """Registry-backed drafts only (not human-reviewed)."""
    from training.review_workflow import template_spec

    drafts = []
    for tid in ("factorial_1", "factorial_3", "palindrome_1"):
        spec = template_spec(tid)
        if not spec.get("problem_statement"):
            continue
        family = tid.rsplit("_", 1)[0]
        drafts.append(
            {
                "spec_id": f"spec.{family}.registry_draft",
                "family_id": family,
                "source_template_ids": [tid],
                "problem_statement": spec["problem_statement"],
                "input_assumptions": [],
                "expected_output_summary": "See bundled template asserts in generator.",
                "boundary_cases": [],
                "provenance": {
                    "source": "generator_registry",
                    "author": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "review_status": "draft",
                    "reviewer_id": None,
                    "notes": spec.get("spec_source"),
                },
                "associated_sample_ids": [],
            }
        )
    return {
        "spec_catalog_version": "v1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_reviewed": False,
        "note": "Draft specs from generator registry excerpts only — not verified by human reviewers.",
        "entries": drafts,
    }
