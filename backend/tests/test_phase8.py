import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import ERROR_LABELS
from backend.app.services.review_service import export_eligibility, build_review_lab_summary
from training.review_workflow import validate_decision_payload, verify_workflow, PRIORITY_SAMPLE_IDS
from training.dataset_validation import sha256_file
from training.label_review_cli import VALID_STATUS

client = TestClient(app)
V3_CHECKSUM = "9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383"
V2_CHECKSUM = "6d2d6ab93918ef86ebbb36d46b0ced5f109dbc0983f2229a067fe72ef194ba55"


def test_v3_checksum_unchanged():
    p = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    if not p.exists():
        pytest.skip("v3 missing")
    assert sha256_file(p) == V3_CHECKSUM


def test_v2_checksum_unchanged():
    p = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"
    if not p.exists():
        pytest.skip("v2 missing")
    assert sha256_file(p) == V2_CHECKSUM


def test_production_checkpoint_unchanged():
    assert (PROJECT_ROOT / "models" / "codebert_classifier" / "config.json").exists()


def test_review_workflow_verify():
    r = verify_workflow()
    assert r["ok"] is True
    assert r["priority_batch_present"] is True


def test_decision_validation_rules():
    ok, _ = validate_decision_payload("rejected", "", None, "alice", False)
    assert ok is False
    ok, _ = validate_decision_payload("relabeled", "typo fix", "correct", "alice", False)
    assert ok is True
    ok, _ = validate_decision_payload("accepted", None, None, None, False)
    assert ok is False
    ok, _ = validate_decision_payload("accepted", None, None, "alice", False)
    assert ok is True
    ok, _ = validate_decision_payload("relabeled", "x", "not_a_label", "alice", False)
    assert ok is False


def test_canonical_labels_only_in_cli():
    assert "relabeled" in VALID_STATUS
    for lbl in ERROR_LABELS:
        ok, _ = validate_decision_payload("relabeled", "r", lbl, "alice", False)
        assert ok is True


def test_export_eligibility_pending_rejected_unresolved():
    train = {"factorial"}
    assert export_eligibility({"status": "pending", "family_id": "palindrome"}, train)["export_eligible"] is False
    assert export_eligibility({"status": "rejected", "family_id": "palindrome", "reviewer_id": "a", "rationale": "x"}, train)[
        "export_eligible"
    ] is False
    assert export_eligibility(
        {"status": "needs_expert_review", "family_id": "palindrome", "reviewer_id": "a", "rationale": "x"}, train
    )["export_eligible"] is False


def test_export_eligibility_accepted_and_relabeled():
    train = {"binary_search"}
    ok = export_eligibility(
        {
            "status": "accepted",
            "family_id": "palindrome",
            "reviewer_id": "alice",
            "rationale": "confirmed",
        },
        train,
    )
    assert ok["export_eligible"] is True
    rel = export_eligibility(
        {
            "status": "relabeled",
            "family_id": "palindrome",
            "reviewer_id": "alice",
            "rationale": "wrong taxonomy",
            "proposed_label": "wrong_operator",
        },
        train,
    )
    assert rel["export_eligible"] is True


def test_export_eligibility_train_family_overlap():
    train = {"factorial"}
    assert export_eligibility(
        {"status": "accepted", "family_id": "factorial", "reviewer_id": "a", "rationale": "ok"},
        train,
    )["export_eligible"] is False


def test_audit_history_append_simulation():
    """Decide path appends prior state before overwrite (no live file mutation)."""
    d = {"review_id": "V3RQ999", "status": "pending", "rationale": None, "reviewer_id": None}
    prior = copy.deepcopy({k: d.get(k) for k in ("status", "rationale", "reviewer_id", "proposed_label")})
    d.setdefault("audit_history", []).append(prior)
    d["status"] = "accepted"
    d["reviewer_id"] = "test_reviewer"
    assert len(d["audit_history"]) == 1
    assert d["audit_history"][0]["status"] == "pending"


def test_review_lab_api_counts_and_blocked():
    r = client.get("/api/evaluation/review-lab?limit=40")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    n = body["counts"]["total_queue"]
    assert n >= 30
    assert body["counts"]["pending"] == n
    assert body["counts"]["adjudicated"] == 0
    assert body["counts"]["export_eligible"] == 0
    assert body["reviewed_eval_slice"]["blocked"] is True
    assert body["reviewed_eval_slice"]["human_reviewed_sample_count"] == 0
    assert body["items"][0]["sample_id"]
    assert body["items"][0]["family_id"]


def test_review_queue_invalid_dataset():
    r = client.get("/api/evaluation/review-queue?dataset=benchmark_v2")
    assert r.status_code == 422


def test_review_packet_api():
    r = client.get("/api/evaluation/review-packet?sample_id=S0027")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["packet"]["sample_id"] == "S0027"
    assert body["synthetic_unreviewed_until_decided"] is True


def test_review_packet_rejects_bad_id():
    assert client.get("/api/evaluation/review-packet?sample_id=../../x").json()["available"] is False
    assert client.get("/api/evaluation/review-packet?review_id=../bad").json()["available"] is False


def test_priority_fixtures_exist():
    for sid in PRIORITY_SAMPLE_IDS:
        fp = PROJECT_ROOT / "data" / "benchmark_v3" / "review_fixtures" / f"{sid}.json"
        assert fp.exists(), sid


def test_review_test_runner_allowlist():
    from training.review_test_runner import _run_tests

    bad = _run_tests("S9999")
    assert bad["ok"] is False
    good = _run_tests("S0003")
    assert good.get("executed") is True
    assert good.get("passed_all") is False


def test_build_slice_stays_blocked():
    from training.build_reviewed_eval_slice import main as build_main

    assert build_main() == 0
    manifest = json.loads(
        (PROJECT_ROOT / "data" / "benchmark_reviewed" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked_pending_human_review"
    assert manifest.get("human_reviewed_sample_count", 0) == 0
    assert manifest.get("next_action")


def test_build_slice_blocked_manifest_idempotent():
    from training.build_reviewed_eval_slice import main as build_main

    build_main()
    m1 = json.loads((PROJECT_ROOT / "data" / "benchmark_reviewed" / "manifest.json").read_text(encoding="utf-8"))
    build_main()
    m2 = json.loads((PROJECT_ROOT / "data" / "benchmark_reviewed" / "manifest.json").read_text(encoding="utf-8"))
    assert m1["status"] == m2["status"] == "blocked_pending_human_review"
    assert m1["human_reviewed_sample_count"] == 0


def test_all_review_items_still_pending():
    data = json.loads(
        (PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json").read_text(encoding="utf-8")
    )
    assert data["human_reviewed"] is False
    assert len(data["decisions"]) >= 30
    assert all(d["status"] == "pending" for d in data["decisions"])
    for d in data["decisions"]:
        assert d.get("original_label") == d.get("current_label")


def test_csv_labels_not_modified_by_review_file():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv")
    decisions = json.loads(
        (PROJECT_ROOT / "data" / "benchmark_v3" / "review_decisions.json").read_text(encoding="utf-8")
    )
    for d in decisions["decisions"]:
        sid = d["sample_id"]
        row = df[df["sample_id"] == sid].iloc[0]
        assert row["error_label"] == d["original_label"]
