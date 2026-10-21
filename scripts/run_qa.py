#!/usr/bin/env python3
"""
CodeInsight AI — unified QA runner.

Usage (from repo root):
  py -3.13 scripts/run_qa.py

Does not start duplicate uvicorn on :8000. Optionally starts Vite when --start-frontend.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_PYTHON = str(VENV_PYTHON if VENV_PYTHON.is_file() else sys.executable)

BUBBLE = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""

BINARY = """def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1"""


@dataclass
class Check:
    name: str
    status: str  # passed, failed, blocked, skipped
    layer: str  # software, api_integration, browser, model_quality
    actual: Any = None
    details: str = ""
    artifact: Optional[str] = None
    measures: str = "software"  # software | model_quality

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QASession:
    started_at: str
    backend_url: str
    frontend_url: str
    checks: List[Check] = field(default_factory=list)
    backend_reused: bool = False
    frontend_reused: bool = False
    results_dir: str = ""

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def counts(self) -> Dict[str, int]:
        out = {"passed": 0, "failed": 0, "blocked": 0, "skipped": 0, "warning": 0}
        for c in self.checks:
            key = c.status if c.status in out else "failed"
            out[key] = out.get(key, 0) + 1
        return out


def http_get_json(url: str, timeout: float = 5.0) -> Dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: Dict, timeout: float = 120.0) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_url(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def run_cmd(
    name: str,
    layer: str,
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Check:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        details = ""
        if not ok:
            details = (proc.stderr or proc.stdout or "")[-4000:]
        return Check(
            name=name,
            status="passed" if ok else "failed",
            layer=layer,
            actual={"exit_code": proc.returncode},
            details=details.strip(),
            measures="software",
        )
    except subprocess.TimeoutExpired as e:
        return Check(name, "failed", layer, details=f"Timeout: {e}", measures="software")
    except FileNotFoundError as e:
        return Check(name, "blocked", layer, details=str(e), measures="software")


def merge_json_checks(path: Path, layer: str, measures: str) -> List[Check]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Check] = []
    for row in data:
        out.append(
            Check(
                name=row.get("name", "unknown"),
                status=row.get("status", "failed"),
                layer=layer,
                actual=row.get("actual"),
                details=row.get("details", ""),
                artifact=row.get("artifact"),
                measures=measures,
            )
        )
    return out


def write_reports(session: QASession, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": session.started_at,
        "backend_url": session.backend_url,
        "frontend_url": session.frontend_url,
        "backend_reused": session.backend_reused,
        "frontend_reused": session.frontend_reused,
        "counts": session.counts(),
        "checks": [c.to_dict() for c in session.checks],
    }
    (out_dir / "qa_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_layer: Dict[str, List[Check]] = {}
    for c in session.checks:
        by_layer.setdefault(c.layer, []).append(c)

    lines = [
        "# CodeInsight AI — QA Summary",
        "",
        f"**Run:** {session.started_at}",
        f"**Backend:** {session.backend_url} ({'reused' if session.backend_reused else 'not detected'})",
        f"**Frontend:** {session.frontend_url} ({'reused' if session.frontend_reused else 'not detected'})",
        "",
        "## Counts",
        "",
    ]
    for k, v in session.counts().items():
        if v:
            lines.append(f"- {k}: {v}")
    lines.append("")

    for layer_title, layer_key in [
        ("Software / build health", "software"),
        ("API integration", "api_integration"),
        ("Browser behavior", "browser"),
        ("Model quality", "model_quality"),
    ]:
        items = [c for c in session.checks if c.layer == layer_key or (layer_key == "software" and c.layer == "software")]
        if layer_key != "software":
            items = [c for c in session.checks if c.layer == layer_key]
        if not items:
            continue
        lines.append(f"## {layer_title}")
        lines.append("")
        for c in items:
            flag = c.status.upper()
            extra = f" — {c.details}" if c.details else ""
            lines.append(f"- **{c.name}**: {flag}{extra}")
        lines.append("")

    software_failed = any(c.status == "failed" and c.measures == "software" for c in session.checks)
    software_blocked = any(c.status == "blocked" and c.layer in ("software", "api_integration", "browser") for c in session.checks)
    mq_warnings = [c for c in session.checks if c.layer == "model_quality" and c.status in ("warning", "failed")]

    lines.append("## Verdict")
    lines.append("")
    if software_failed:
        lines.append("**Software QA: FAILED** (fix failures before release).")
    elif software_blocked:
        lines.append("**Software QA: INCOMPLETE** (required checks blocked).")
    else:
        lines.append("**Software QA: PASSED** (infrastructure and contracts).")
    if mq_warnings:
        lines.append("")
        lines.append("**Model quality:** see warnings above — weak classifier metrics are expected; similarity test set is small.")
    (out_dir / "qa_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CodeInsight AI QA suite")
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:5173")
    parser.add_argument("--start-frontend", action="store_true", help="Start Vite if :5173 is down")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument("--skip-model-quality", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.results_dir or (PROJECT_ROOT / "test-results" / f"qa_{ts}")
    session = QASession(
        started_at=datetime.now(timezone.utc).isoformat(),
        backend_url=args.backend.rstrip("/"),
        frontend_url=args.frontend.rstrip("/"),
        results_dir=str(out_dir),
    )

    frontend_proc: Optional[subprocess.Popen] = None
    we_started_frontend = False

    # 1 — Environment
    py = DEFAULT_PYTHON
    session.add(
        Check(
            "env_python",
            "passed" if Path(py).exists() or shutil.which(py) else "blocked",
            "software",
            actual={"python": py, "version": sys.version.split()[0]},
        )
    )
    node = shutil.which("node")
    npm = shutil.which("npm")
    session.add(
        Check(
            "env_node",
            "passed" if node else "blocked",
            "software",
            actual={"node": node, "npm": npm},
            details="" if node else "Node.js not found",
        )
    )
    if node:
        nv = subprocess.run([node, "--version"], capture_output=True, text=True)
        session.checks[-1].actual["node_version"] = (nv.stdout or nv.stderr).strip()

    req = PROJECT_ROOT / "backend" / "requirements.txt"
    session.add(
        Check(
            "env_backend_requirements",
            "passed" if req.is_file() else "blocked",
            "software",
            actual=str(req.relative_to(PROJECT_ROOT)),
        )
    )

    # 2 — Backend health (reuse only; never start/stop uvicorn here)
    backend_up = probe_url(f"{session.backend_url}/api/health")
    if backend_up:
        session.backend_reused = True
        try:
            health = http_get_json(f"{session.backend_url}/api/health")
            session.add(
                Check(
                    "backend_health",
                    "passed" if health.get("status") == "ok" else "failed",
                    "software",
                    actual=health,
                )
            )
            ready = http_get_json(f"{session.backend_url}/api/ready")
            session.add(
                Check(
                    "backend_ready",
                    "passed" if ready.get("ready") else "warning",
                    "software",
                    actual=ready,
                    details="" if ready.get("ready") else "Dataset or models not fully ready",
                )
            )
            live = http_post_json(
                f"{session.backend_url}/api/similarity/analyze",
                {"code_a": BUBBLE, "code_b": BINARY, "threshold": 0.8},
            )
            hybrid_ok = live.get("is_similar") is False and "hybrid" in (live.get("decision_model") or "")
            session.add(
                Check(
                    "backend_serves_hybrid_policy",
                    "passed" if hybrid_ok else "failed",
                    "software",
                    actual={
                        "decision_model": live.get("decision_model"),
                        "decision_policy": live.get("decision_policy"),
                        "is_similar": live.get("is_similar"),
                    },
                    details="Stale backend may show codebert-only decisions; restart project uvicorn if this fails.",
                )
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            session.add(Check("backend_health", "failed", "software", details=str(e)))
    else:
        session.add(
            Check(
                "backend_health",
                "blocked",
                "software",
                details=f"No server at {session.backend_url}. Start uvicorn once; runner will not spawn a duplicate.",
            )
        )

    # 3 — Pytest
    pytest_cmd = [py, "-m", "pytest", "backend/tests", "-q", "--tb=no"]
    session.add(run_cmd("backend_pytest", "software", pytest_cmd, timeout=600))

    # 4 — Frontend build
    if npm:
        session.add(
            run_cmd(
                "frontend_production_build",
                "software",
                [npm, "run", "build"],
                cwd=PROJECT_ROOT / "frontend",
                timeout=300,
            )
        )
    else:
        session.add(Check("frontend_production_build", "blocked", "software", details="npm missing"))

    # 5 — Live API integration (Node)
    api_report = out_dir / "api_integration.json"
    if node and backend_up:
        env = os.environ.copy()
        env["BACKEND_URL"] = session.backend_url
        env["QA_REPORT_PATH"] = str(api_report)
        proc = subprocess.run(
            [node, "scripts/api-integration-qa.mjs"],
            cwd=PROJECT_ROOT / "frontend",
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        session.checks.extend(merge_json_checks(api_report, "api_integration", "software"))
        if proc.returncode != 0 and not api_report.is_file():
            session.add(
                Check(
                    "api_integration_runner",
                    "failed" if proc.returncode == 1 else "blocked",
                    "api_integration",
                    details=(proc.stderr or proc.stdout)[-3000:],
                )
            )
    else:
        session.add(
            Check(
                "api_integration_runner",
                "blocked",
                "api_integration",
                details="Node or backend unavailable",
            )
        )

    # 6 — Browser E2E
    frontend_up = probe_url(session.frontend_url)
    if not frontend_up and args.start_frontend and npm:
        frontend_proc = subprocess.Popen(
            [npm, "run", "dev"],
            cwd=PROJECT_ROOT / "frontend",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        we_started_frontend = True
        for _ in range(40):
            if probe_url(session.frontend_url):
                frontend_up = True
                break
            time.sleep(0.5)

    if frontend_up:
        session.frontend_reused = not we_started_frontend
    elif not args.skip_browser:
        session.add(
            Check(
                "browser_suite",
                "blocked",
                "browser",
                details=f"{session.frontend_url} not reachable. Run `npm run dev` or pass --start-frontend.",
            )
        )

    if frontend_up and not args.skip_browser and node:
        base = session.frontend_url
        screenshot_dir = out_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        common_env = {
            **os.environ,
            "BASE_URL": base,
            "QA_SCREENSHOT_DIR": str(screenshot_dir),
        }
        for script, label in [
            ("scripts/e2e-smoke.mjs", "browser_e2e_smoke"),
            ("scripts/similarity-browser-qa.mjs", "browser_similarity_qa"),
            ("scripts/classifier-browser-qa.mjs", "browser_classifier_qa"),
        ]:
            proc = subprocess.run(
                [node, script],
                cwd=PROJECT_ROOT / "frontend",
                env=common_env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            session.add(
                Check(
                    label,
                    "passed" if proc.returncode == 0 else "failed",
                    "browser",
                    actual={"exit_code": proc.returncode},
                    details="" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-3000:],
                    artifact=str(screenshot_dir) if proc.returncode != 0 else None,
                )
            )
    elif args.skip_browser:
        session.add(Check("browser_suite", "skipped", "browser", details="--skip-browser"))

    # 7 — Model quality
    if not args.skip_model_quality:
        mq_report = out_dir / "model_quality.json"
        mq_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "qa_model_quality.py"),
            "--backend",
            session.backend_url,
            "--report",
            str(mq_report),
        ]
        if not backend_up:
            mq_cmd.append("--no-live")
        proc = subprocess.run(mq_cmd, capture_output=True, text=True, timeout=600)
        session.checks.extend(merge_json_checks(mq_report, "model_quality", "model_quality"))
        if proc.returncode == 1:
            session.add(
                Check(
                    "model_quality_gate",
                    "failed",
                    "model_quality",
                    details="One or more model-quality checks failed (live policy probe).",
                )
            )
        elif proc.returncode == 2 and not mq_report.is_file():
            session.add(Check("model_quality_gate", "blocked", "model_quality", details=proc.stderr))

    write_reports(session, out_dir)

    if we_started_frontend and frontend_proc:
        frontend_proc.terminate()
        try:
            frontend_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            frontend_proc.kill()

    counts = session.counts()
    print(textwrap.dedent(f"""
        QA complete -> {out_dir}
        passed={counts['passed']} failed={counts['failed']} blocked={counts['blocked']} skipped={counts['skipped']} warning={counts.get('warning', 0)}
    """).strip())

    software_bad = any(
        c.status in ("failed", "blocked")
        for c in session.checks
        if c.layer in ("software", "api_integration", "browser") and c.status != "skipped"
    )
    return 1 if software_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
