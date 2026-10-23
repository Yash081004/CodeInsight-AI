"""Phase 14 — mutation IDs, queue provenance, v4 eligibility dimensions."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_stable_mutation_id_deterministic():
    from training.mutation_identity import stable_mutation_id

    a = stable_mutation_id(
        template_id="fibonacci_1",
        error_label="wrong_condition",
        mutation_type="boundary_condition",
        code_hash="abc" * 21 + "a",
    )
    b = stable_mutation_id(
        template_id="fibonacci_1",
        error_label="wrong_condition",
        mutation_type="boundary_condition",
        code_hash="abc" * 21 + "a",
    )
    assert a == b
    assert a.startswith("v1.fibonacci_1.")


def test_mutation_registry_links_queue_and_hashes():
    from training.queue_mutation_review import build_mutation_registry
    from training.mutation_identity import lookup_by_id

    reg = build_mutation_registry()
    assert reg["record_count"] >= 10
    first = reg["records"][0]
    assert lookup_by_id(reg, first["mutation_id"]) == first
    assert "code_hash_sha256" in first
    assert "bundled_tests_pass" in first


def test_queue_decisions_template_not_auto_decided():
    path = PROJECT_ROOT / "data/benchmark_v3/queue_mutation_decisions.json"
    if not path.is_file():
        subprocess.run([sys.executable, "-m", "training.phase14_run"], cwd=PROJECT_ROOT, check=True, timeout=120)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["human_reviewed"] is False
    assert all(d["status"] == "pending" for d in data["decisions"])


def test_approved_spec_does_not_clear_label_concern():
    from training.phase14_v4_eligibility import assess_row
    from training.phase10_v3_exclusions import load_review_queue_template_ids
    from training.spec_catalog import load_catalog
    from training.benchmark_v4_build import human_reviewed_families

    row = {
        "sample_id": "S0014",
        "source_template_id": "fibonacci_1",
        "family_id": "fibonacci",
        "error_label": "wrong_condition",
        "mutation_type": "boundary_condition",
        "expected_behavior": "fails_tests",
        "code": "def fibonacci(n):\n    pass\n",
    }
    queue_tids = load_review_queue_template_ids(
        PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
    )
    # Simulate spec approved for family but queue template + pending review remain
    reviewed = human_reviewed_families(load_catalog()) | {"fibonacci"}
    pr = assess_row(row, queue_tids=queue_tids, reviewed_fams=reviewed, pending_samples={"S0014"})
    assert pr["approved_spec_does_not_clear_label_concern"] is True
    assert pr["strict_v4_eligible"] is False


def test_v4_strict_zero_without_human_specs():
    from training.phase14_v4_eligibility import build_eligibility_report

    report = build_eligibility_report(write_csv=False)
    assert report["strict_v4_eligible_count"] == 0
    assert report["v4_ready_for_training"] is False


def test_phase14_run():
    proc = subprocess.run(
        [sys.executable, "-m", "training.phase14_run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (PROJECT_ROOT / "artifacts/metrics/phase14/mutation_registry.json").is_file()
