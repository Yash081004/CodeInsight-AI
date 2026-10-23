"""Phase 9 read-only dataset audit tooling."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_phase9_audit_generates_report():
    proc = subprocess.run(
        [sys.executable, "-m", "training.phase9_dataset_audit"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    report = PROJECT_ROOT / "artifacts/metrics/phase9/dataset_audit_report.json"
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "datasets" in data
    assert "legacy_v1" in data["datasets"]
    v1 = data["datasets"]["legacy_v1"]
    assert v1["checksum_sha256"] == "c7ae7a8d836b64c9aa139e9c1fe77724329840a5629f87a08b3a2d1e993aff37"
    assert v1["split_overlap"]["family_leakage"] is False
