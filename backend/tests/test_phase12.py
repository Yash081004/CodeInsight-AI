"""Phase 12 schema, audit, and triage (no training)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_spec_catalog_validates_draft():
    from training.spec_catalog import build_initial_draft_catalog, validate_catalog

    cat = build_initial_draft_catalog()
    ok, errs = validate_catalog(cat)
    assert ok, errs
    assert cat["human_reviewed"] is False
    assert all(e["provenance"]["review_status"] == "draft" for e in cat["entries"])


def test_legacy_row_without_catalog():
    from training.spec_catalog import legacy_row_has_spec, load_catalog

    row = {"family_id": "nonexistent_family_xyz", "expected_behavior": "passes_all_tests"}
    assert legacy_row_has_spec(row, load_catalog()) is False


def test_phase12_audit_produces_json():
    proc = subprocess.run(
        [sys.executable, "-m", "training.phase12_run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    audit = PROJECT_ROOT / "artifacts/metrics/phase12/dataset_audit.json"
    triage = PROJECT_ROOT / "artifacts/metrics/phase12/review_triage.json"
    catalog = PROJECT_ROOT / "data/specifications/catalog.v1.json"
    assert audit.is_file() and triage.is_file() and catalog.is_file()
    data = json.loads(triage.read_text(encoding="utf-8"))
    assert data["human_reviewed"] is False
    assert len(data["items"]) == 10


def test_review_triage_no_human_decisions():
    triage = json.loads(
        (PROJECT_ROOT / "artifacts/metrics/phase12/review_triage.json").read_text(encoding="utf-8")
    )
    for item in triage["items"]:
        assert item.get("human_reviewed") is False
        assert item["recommended_next_action"] in {
            "mechanically_verifiable",
            "requires_reviewer_intended_behavior",
            "requires_expert_decision",
            "insufficient_evidence",
        }


def test_spec_code_separator_unchanged_from_phase11():
    from training.spec_code_input import build_model_input

    row = {"code": "def f(): pass", "problem_type": "gcd"}
    assert "[SEP]" in build_model_input(row, "spec_code")
