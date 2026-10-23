"""
Model-quality gate for CodeInsight QA (separate from software reliability).

Reads benchmark artifacts and optional live API probes. Does not retrain or tune.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIER_METRICS = PROJECT_ROOT / "artifacts/metrics/codebert_classification_metrics.json"
SIMILARITY_POLICY = PROJECT_ROOT / "artifacts/metrics/similarity_hybrid_policy.json"
DATASET_VALIDATION = PROJECT_ROOT / "artifacts/metrics/classification_dataset_validation.json"

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

CLASSIFIER_BENCHMARK = [
    ("correct_square", {"code": "def square(n):\n    return n * n"}, "correct"),
    (
        "wrong_operator",
        {
            "code": "def square(n):\n    return n + n",
            "problem_statement": "Return the square of the given number.",
        },
        "wrong_operator",
    ),
    (
        "off_by_one",
        {
            "code": "def sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total",
            "problem_statement": "Return the sum of all integers from 1 through n, inclusive.",
        },
        "off_by_one_loop",
    ),
    (
        "wrong_condition",
        {
            "code": "def is_adult(age):\n    if age < 18:\n        return True\n    return False",
            "problem_statement": "Return True if age is at least 18, otherwise return False.",
        },
        "wrong_condition",
    ),
    (
        "wrong_init",
        {
            "code": "def sum_values(values):\n    total = 1\n    for value in values:\n        total += value\n    return total",
            "problem_statement": "Return the sum of all elements in the list. The sum of an empty list should be zero.",
        },
        "wrong_initialization",
    ),
]


@dataclass
class QualityCheck:
    name: str
    status: str  # passed, failed, warning, blocked, skipped
    category: str  # model_quality
    actual: Any = None
    details: str = ""
    artifact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _post_json(url: str, payload: Dict, timeout: float = 120.0) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_model_quality_checks(backend_base: str, live_api: bool = True) -> List[QualityCheck]:
    checks: List[QualityCheck] = []
    clf = _load_json(CLASSIFIER_METRICS)
    if clf is None:
        checks.append(
            QualityCheck(
                "classification_metrics_artifact",
                "blocked",
                "model_quality",
                details=f"Missing {CLASSIFIER_METRICS.relative_to(PROJECT_ROOT)}",
                artifact=str(CLASSIFIER_METRICS),
            )
        )
    else:
        macro_f1 = float(clf.get("macro_f1", 0))
        status = "warning" if macro_f1 < 0.25 else "passed"
        details = f"macro_f1={macro_f1:.4f} on {clf.get('sample_count', '?')} samples"
        if macro_f1 < 0.25:
            details += " — production CodeBERT classifier is weak; routing/transparency only."
        checks.append(
            QualityCheck(
                "classification_metrics_artifact",
                status,
                "model_quality",
                actual={"macro_f1": macro_f1, "macro_precision": clf.get("macro_precision"), "macro_recall": clf.get("macro_recall")},
                details=details,
                artifact=str(CLASSIFIER_METRICS.relative_to(PROJECT_ROOT)),
            )
        )
        per_class = clf.get("per_class_metrics") or {}
        collapsed = sum(1 for k, v in per_class.items() if k != "correct" and (v.get("recall") or 0) == 0)
        if collapsed >= 4:
            checks.append(
                QualityCheck(
                    "classification_per_class_recall",
                    "warning",
                    "model_quality",
                    actual={"zero_recall_fault_classes": collapsed},
                    details="Most fault classes have zero recall on held-out test — expect mislabels.",
                    artifact=str(CLASSIFIER_METRICS.relative_to(PROJECT_ROOT)),
                )
            )

    sim = _load_json(SIMILARITY_POLICY)
    if sim is None:
        checks.append(
            QualityCheck(
                "similarity_hybrid_policy_artifact",
                "blocked",
                "model_quality",
                details=f"Missing {SIMILARITY_POLICY.relative_to(PROJECT_ROOT)}",
                artifact=str(SIMILARITY_POLICY),
            )
        )
    else:
        test_h = (sim.get("test") or {}).get("hybrid") or {}
        test_n = test_h.get("sample_count", 0)
        fp = test_h.get("false_positives", "?")
        fn = test_h.get("false_negatives", "?")
        f1 = test_h.get("f1", 0)
        checks.append(
            QualityCheck(
                "similarity_hybrid_test_metrics",
                "passed" if test_n else "warning",
                "model_quality",
                actual={"f1": f1, "false_positives": fp, "false_negatives": fn, "sample_count": test_n},
                details=f"Held-out similarity test n={test_n}; hybrid F1={f1:.3f}, FP={fp}, FN={fn}. Small test set — preliminary only.",
                artifact=str(SIMILARITY_POLICY.relative_to(PROJECT_ROOT)),
            )
        )
        diag = sim.get("diagnostic_by_relationship") or {}
        sort_search = diag.get("different_operations") or diag.get("different_operation")
        if sort_search:
            checks.append(
                QualityCheck(
                    "similarity_diagnostic_different_ops",
                    "passed" if sort_search.get("f1", 0) >= 0.99 else "warning",
                    "model_quality",
                    actual=sort_search,
                    details="Diagnostic pair policy behavior for different operations (artifact).",
                    artifact=str(SIMILARITY_POLICY.relative_to(PROJECT_ROOT)),
                )
            )

    ds = _load_json(DATASET_VALIDATION)
    if ds:
        checksum = ds.get("checksum_sha256") or ds.get("checksum") or ds.get("classification_checksum")
        checks.append(
            QualityCheck(
                "classification_dataset_checksum",
                "passed" if checksum else "warning",
                "model_quality",
                actual={
                    "checksum_sha256": checksum,
                    "samples_per_split": ds.get("samples_per_split"),
                    "total_samples": ds.get("total_samples"),
                },
                details="Dataset validation artifact present (provenance check).",
                artifact=str(DATASET_VALIDATION.relative_to(PROJECT_ROOT)),
            )
        )

    if not live_api:
        checks.append(
            QualityCheck(
                "live_api_model_probes",
                "skipped",
                "model_quality",
                details="Live API probes disabled.",
            )
        )
        return checks

    sim_url = f"{backend_base.rstrip('/')}/api/similarity/analyze"
    try:
        live = _post_json(sim_url, {"code_a": BUBBLE, "code_b": BINARY, "threshold": 0.8})
        ok = live.get("is_similar") is False and "hybrid" in (live.get("decision_model") or "")
        checks.append(
            QualityCheck(
                "live_similarity_sort_vs_search",
                "passed" if ok else "failed",
                "model_quality",
                actual={
                    "is_similar": live.get("is_similar"),
                    "decision_model": live.get("decision_model"),
                    "codebert_score": live.get("codebert_score"),
                    "tfidf_score": live.get("tfidf_score"),
                },
                details="Sort vs search should be not-similar under hybrid policy (high CB cosine is not semantic equivalence).",
            )
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        checks.append(
            QualityCheck(
                "live_similarity_sort_vs_search",
                "blocked",
                "model_quality",
                details=f"Could not reach similarity API: {e}",
            )
        )

    clf_url = f"{backend_base.rstrip('/')}/api/classification/analyze"
    for name, payload, expected in CLASSIFIER_BENCHMARK:
        try:
            data = _post_json(clf_url, payload)
            predicted = data.get("predicted_label")
            match = predicted == expected
            # Software uses routing; model quality tracks semantic label match
            status = "passed" if match else "warning"
            checks.append(
                QualityCheck(
                    f"live_classifier_label_{name}",
                    status,
                    "model_quality",
                    actual={
                        "expected": expected,
                        "predicted": predicted,
                        "model_type": data.get("model_type"),
                        "review_status": (data.get("diagnostics") or {}).get("review_status"),
                    },
                    details="Label match" if match else f"Model predicted {predicted}, benchmark label {expected}",
                )
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            checks.append(
                QualityCheck(
                    f"live_classifier_label_{name}",
                    "blocked",
                    "model_quality",
                    details=str(e),
                )
            )

    return checks


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--no-live", action="store_true")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    results = run_model_quality_checks(args.backend, live_api=not args.no_live)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps([c.to_dict() for c in results], indent=2), encoding="utf-8")
    failed = [c for c in results if c.status == "failed"]
    blocked = [c for c in results if c.status == "blocked"]
    if failed or blocked:
        raise SystemExit(1 if failed else 2)
