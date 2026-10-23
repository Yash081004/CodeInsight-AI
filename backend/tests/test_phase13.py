"""Phase 13 — queue reconciliation, spec packets, benchmark v4 (no training)."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_queue_reconciliation_all_ten_items():
    from training.phase13_queue_reconciliation import build_reconciliation_report

    report = build_reconciliation_report()
    assert len(report["items"]) == 10
    classes = {i["mismatch_class"] for i in report["items"]}
    assert "queue_only_excluded_mutation" in classes
    assert report["human_reviewed"] is False


def test_insufficient_evidence_items_are_queue_only_excluded():
    """Phase 12 'insufficient_evidence' queue rows are excluded-at-generation mutations."""
    from training.phase13_queue_reconciliation import build_reconciliation_report

    report = build_reconciliation_report()
    queue_only = [
        i
        for i in report["items"]
        if i["mismatch_class"] == "queue_only_excluded_mutation"
    ]
    assert len(queue_only) == 5


def test_authoring_packets_merge_as_draft():
    from training.phase13_spec_packets import load_authoring_packets, merge_packets_into_catalog
    from training.spec_catalog import validate_catalog

    packets = load_authoring_packets()
    assert len(packets) >= 5
    assert all(p.get("human_reviewed") is False for p in packets)
    catalog, errs = merge_packets_into_catalog(write=False)
    assert not errs, errs
    ok, val_errs = validate_catalog(catalog)
    assert ok, val_errs
    fams = {e["family_id"] for e in catalog["entries"]}
    for fam in ("palindrome", "fibonacci", "linear_search", "factorial", "gcd"):
        assert fam in fams
        ent = next(e for e in catalog["entries"] if e["family_id"] == fam)
        assert ent["provenance"]["review_status"] == "draft"


def test_benchmark_v4_build_reproducible():
    from training.benchmark_v4_build import build_v4

    m1 = build_v4(write_csv=False)
    m2 = build_v4(write_csv=False)
    assert m1["source_benchmark_v3_csv_checksum_sha256"] == m2["source_benchmark_v3_csv_checksum_sha256"]
    assert m1["eligible_row_count"] == m2["eligible_row_count"]
    assert m1["family_split_isolation_eligible"]["ok"] is True


def test_benchmark_v4_quarantines_without_human_specs():
    from training.benchmark_v4_build import build_v4

    manifest = build_v4(write_csv=False)
    assert manifest["eligible_row_count"] == 0
    assert manifest["quarantine_row_count"] == 190
    assert manifest["labels_unchanged_from_v3"] is True


def test_phase13_run_orchestrator():
    proc = subprocess.run(
        [sys.executable, "-m", "training.phase13_run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (PROJECT_ROOT / "docs/phase13/QUEUE_MISMATCH_REPORT.md").is_file()
    assert (PROJECT_ROOT / "data/benchmark_v4/manifest.json").is_file()
