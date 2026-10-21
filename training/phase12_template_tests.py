"""
Run fixed generator template tests against dataset sample code (bounded subprocess).

Only uses test strings from training.generate_dataset.TEMPLATES registry — not user/API code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _template_test_map() -> Dict[str, str]:
    import training.generate_dataset as gd

    return {t["template_id"]: t["test_code"] for t in gd.TEMPLATES}


def run_template_tests_on_code(template_id: str, code: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    tests = _template_test_map().get(template_id)
    if not tests:
        return {
            "template_id": template_id,
            "executed": False,
            "reason": "no_registry_tests",
        }
    script = code.strip() + "\n\n" + tests.strip() + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "template_id": template_id,
            "executed": True,
            "exit_code": proc.returncode,
            "passed_all": proc.returncode == 0,
            "stderr_preview": (proc.stderr or "")[:400],
            "note": "Bundled generator tests only — passing does not prove label correct.",
        }
    except subprocess.TimeoutExpired:
        return {"template_id": template_id, "executed": False, "error": "timeout"}
    except Exception as e:
        return {"template_id": template_id, "executed": False, "error": str(e)}
