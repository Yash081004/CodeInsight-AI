import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.config import ERROR_LABELS
from backend.app.services.review_service import audit_integrity, load_v3_review_decisions
from backend.app.services.task_context_service import assess_task_context
from training.dataset_validation import sha256_file

client = TestClient(app)

V2_CHECKSUM = "6d2d6ab93918ef86ebbb36d46b0ced5f109dbc0983f2229a067fe72ef194ba55"
V3_CHECKSUM = "9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383"


def test_v2_checksum_preserved():
    p = PROJECT_ROOT / "data" / "benchmark_v2" / "classification" / "dataset.csv"
    if not p.exists():
        pytest.skip("v2 missing")
    assert sha256_file(p) == V2_CHECKSUM


def test_v3_checksum_preserved():
    p = PROJECT_ROOT / "data" / "benchmark_v3" / "classification" / "dataset.csv"
    if not p.exists():
        pytest.skip("v3 missing")
    assert sha256_file(p) == V3_CHECKSUM


def test_production_checkpoint_path_unchanged():
    prod = PROJECT_ROOT / "models" / "codebert_classifier" / "config.json"
    assert prod.exists()


def test_task_context_flags_missing_spec():
    ctx = assess_task_context("wrong_condition", code="def f(n): return n % 2 == 1")
    assert ctx["requires_external_context"] is True
    assert "no_problem_statement_provided" in ctx["limitations"]


def test_classification_accepts_optional_problem_statement():
    r = client.post(
        "/api/classification/analyze",
        json={"code": "def add(a,b): return a+b", "problem_statement": "Return sum of two ints"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("task_context") is not None
    assert body.get("tests_executed") is False


def test_phase7_error_analysis_endpoint():
    report_path = PROJECT_ROOT / "artifacts" / "metrics" / "phase7" / "error_analysis_report.json"
    if not report_path.exists():
        pytest.skip("run phase7_error_analysis first")
    r = client.get("/api/evaluation/phase7-error-analysis")
    assert r.status_code == 200
    assert r.json()["available"] is True


def test_error_explorer_rejects_bad_family_id():
    r = client.get("/api/evaluation/error-explorer?family_id=../etc")
    assert r.json()["available"] is False


def test_review_decisions_integrity():
    data = load_v3_review_decisions()
    if not data.get("available"):
        pytest.skip("v3 review decisions not generated")
    assert data.get("human_reviewed") is False
    assert all(d.get("status") == "pending" for d in data.get("decisions", []))
    assert audit_integrity() == []


def test_reviewed_slice_blocked():
    r = client.get("/api/evaluation/review-queue")
    assert r.status_code == 200
    body = r.json()
    if body.get("available"):
        assert body.get("adjudication_complete") is False
        assert body["reviewed_eval_slice"].get("human_reviewed_sample_count", 0) == 0


def test_build_reviewed_slice_blocked_without_decisions():
    from training.build_reviewed_eval_slice import main as build_main

    rc = build_main()
    assert rc in (0, 2)
