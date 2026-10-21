"""
Execute allowlisted review behavioral tests against benchmark sample code.

Isolation: subprocess, 2s timeout, no network hook (best-effort on Windows).
Does NOT execute arbitrary user/API code — only PRIORITY_SAMPLE_IDS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.review_workflow import PRIORITY_SAMPLE_IDS, load_sample_row


def _run_tests(sample_id: str) -> dict:
    if sample_id not in PRIORITY_SAMPLE_IDS:
        return {"ok": False, "error": "sample_id not allowlisted for review test execution"}
    row = load_sample_row(sample_id)
    if not row:
        return {"ok": False, "error": "sample not found"}
    fixture_path = PROJECT_ROOT / "data" / "benchmark_v3" / "review_fixtures" / f"{sample_id}.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.exists() else {}
    tests = fixture.get("review_only_tests", [])
    code = row["code"]
    script = code + "\n\n"
    for t in tests:
        script += t["code"] + "\n"
    wrapped = script
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "sample_id": sample_id,
            "executed": True,
            "exit_code": proc.returncode,
            "passed_all": proc.returncode == 0,
            "stderr": (proc.stderr or "")[:500],
            "note": "Result covers listed asserts only — not universal correctness or label proof.",
        }
    except subprocess.TimeoutExpired:
        return {"sample_id": sample_id, "executed": False, "error": "timeout_2s"}
    except Exception as e:
        return {"sample_id": sample_id, "executed": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--all-priority", action="store_true")
    args = parser.parse_args()
    if args.all_priority:
        out = [_run_tests(sid) for sid in sorted(PRIORITY_SAMPLE_IDS)]
        print(json.dumps(out, indent=2))
        return 0
    if not args.sample_id:
        print(json.dumps({"ok": False, "error": "provide --sample-id or --all-priority"}, indent=2))
        return 1
    print(json.dumps(_run_tests(args.sample_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
