"""
Phase 14 orchestrator — review readiness, mutation registry, v4 eligibility, docs.

Usage: py -3.13 -m training.phase14_run
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.benchmark_v4_build import build_v4
from training.phase14_v4_eligibility import build_eligibility_report
from training.queue_mutation_review import (
    build_mutation_registry,
    load_queue_mutation_decisions,
    write_registry_artifact,
)


def _preserve_reconciliation() -> None:
    src = PROJECT_ROOT / "artifacts/metrics/phase13/queue_reconciliation.json"
    dst_dir = PROJECT_ROOT / "artifacts/metrics/phase14"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst_dir / "queue_reconciliation_preserved.json")


def _docs_dir() -> Path:
    d = PROJECT_ROOT / "docs/phase14"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_review_workflow_doc() -> None:
    text = """# Phase 14 — Review workflow commands

## Sample-based review (existing CSV rows)

```powershell
py -3.13 -m training.label_review_cli --dataset benchmark_v3 list
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --id V3RQ001
py -3.13 -m training.label_review_cli --dataset benchmark_v3 verify
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --reviewer alice
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --anonymous-review
```

Decisions append `audit_history` in `data/benchmark_v3/review_decisions.json`. **dataset.csv is not modified.**

## Queue mutation review (Batch A/B — no CSV sample_id)

Stable IDs: `artifacts/metrics/phase14/mutation_registry.json` (`mutation_id` field).

```powershell
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue list
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue packet --id QMRQ001
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue decide --id QMRQ001 --status needs_expert_review --rationale "..." --reviewer alice
```

Decisions: `data/benchmark_v3/queue_mutation_decisions.json`.

### Batch mapping

| Batch | QMRQ ids (see `queue list`) | Phase 13 class |
|-------|----------------------------|----------------|
| A | Queue-only excluded mutations | `queue_only_excluded_mutation` |
| B | CSV-linked / taxonomy / stale reason | collision or stale |

## Specification approval (draft → human_reviewed)

```powershell
py -3.13 -m training.spec_approval_cli list
py -3.13 -m training.spec_approval_cli checklist --family palindrome
py -3.13 -m training.spec_approval_cli approve --family palindrome --reviewer NAME --notes "..."
```

**Do not run `approve` until a real reviewer resolves checklist ambiguities.**

## Evidence semantics

| Observation | Does NOT imply |
|-------------|----------------|
| Bundled tests fail | Label is correct |
| Bundled tests pass | Code is logically correct |
| Approved family spec | Every sample label in family is correct |
"""
    (_docs_dir() / "REVIEW_WORKFLOW.md").write_text(text, encoding="utf-8")


def write_go_no_go(elig: dict) -> None:
    ready = elig.get("v4_ready_for_training") and elig.get("strict_v4_eligible_count", 0) > 0
    text = f"""# Phase 14 — Go / no-go for model training

**Decision: NO-GO**

- Strict v4 eligible rows: **{elig.get("strict_v4_eligible_count", 0)}** / 190
- v4 ready gate: **{ready}**
- Queue mutation adjudication: pending (`queue_mutation_decisions.json`)
- Specification approvals: all five families remain **draft**

Training must not proceed until strict eligibility passes and human review completes.
"""
    (_docs_dir() / "GO_NO_GO.md").write_text(text, encoding="utf-8")


def write_v4_eligibility_doc(elig: dict) -> None:
    lines = [
        "# Phase 14 — Benchmark v4 eligibility reassessment",
        "",
        f"Strict v4 eligible: **{elig.get('strict_v4_eligible_count')}** / 190",
        f"Narrow (except spec): **{elig.get('narrow_protocol_eligible_except_spec_count')}** rows would remain blocked by missing human-reviewed spec.",
        "",
        "## Independent quarantine reason counts",
        "",
        "```json",
        json.dumps(elig.get("independent_quarantine_reason_counts"), indent=2),
        "```",
        "",
        "## Protocol",
        "",
        "- **Strict:** spec + no review-queue template + no style variant + correct-row policy + no pending sample review.",
        "- **Narrow (documented only):** fault rows with `fails_tests` and failing bundled tests — still **not** promoted without human-reviewed spec.",
        "- **Spec ≠ label:** approved specification does not clear queue-template or pending V3RQ label review.",
        "",
        "Machine-readable: `artifacts/metrics/phase14/v4_eligibility_assessment.json`",
    ]
    (_docs_dir() / "V4_ELIGIBILITY.md").write_text("\n".join(lines), encoding="utf-8")


def write_spec_approval_doc() -> None:
    text = """# Phase 14 — Specification approval (pending human)

All five priority families remain **`draft`** / **`human_reviewed: false`**.

| Family | Checklist command |
|--------|-------------------|
| palindrome | `py -3.13 -m training.spec_approval_cli checklist --family palindrome` |
| fibonacci | `py -3.13 -m training.spec_approval_cli checklist --family fibonacci` |
| linear_search | `py -3.13 -m training.spec_approval_cli checklist --family linear_search` |
| factorial | `py -3.13 -m training.spec_approval_cli checklist --family factorial` |
| gcd | `py -3.13 -m training.spec_approval_cli checklist --family gcd` |

Approval (human only):

`py -3.13 -m training.spec_approval_cli approve --family <id> --reviewer <NAME> --notes "..."`

Recorded ambiguities appear in checklist JSON — resolve or explicitly accept before approve.
"""
    (_docs_dir() / "SPEC_APPROVAL.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    (_docs_dir() / "README.md").write_text(
        """# Phase 14 — Audited human review & benchmark v4 readiness

```powershell
py -3.13 -m training.phase14_run
```

| Deliverable | Path |
|-------------|------|
| Review commands | [REVIEW_WORKFLOW.md](./REVIEW_WORKFLOW.md) |
| Mutation registry | `artifacts/metrics/phase14/mutation_registry.json` |
| v4 eligibility | `artifacts/metrics/phase14/v4_eligibility_assessment.json` |
| Preserved reconciliation | `artifacts/metrics/phase14/queue_reconciliation_preserved.json` |
| Go/no-go | [GO_NO_GO.md](./GO_NO_GO.md) |
""",
        encoding="utf-8",
    )


def main() -> int:
    _preserve_reconciliation()
    write_registry_artifact()
    load_queue_mutation_decisions(create_if_missing=True)
    reg = build_mutation_registry()
    elig = build_eligibility_report()
    manifest = build_v4(write_csv=True)
    manifest_path = PROJECT_ROOT / "artifacts/metrics/phase14/benchmark_v4_build.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_review_workflow_doc()
    write_v4_eligibility_doc(elig)
    write_spec_approval_doc()
    write_go_no_go(elig)
    write_readme()

    summary = {
        "mutation_registry_records": reg["record_count"],
        "strict_v4_eligible": elig["strict_v4_eligible_count"],
        "narrow_except_spec": elig["narrow_protocol_eligible_except_spec_count"],
    }
    (PROJECT_ROOT / "artifacts/metrics/phase14/phase14_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Phase 14 complete.", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
