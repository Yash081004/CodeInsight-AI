"""
Phase 12 orchestrator — audit, triage, spec catalog bootstrap, docs.

Usage: py -3.13 -m training.phase12_run
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.phase12_dataset_audit import run_all as run_audit
from training.phase12_review_triage import QUEUE_PATH, build_prioritized_packet, triage_queue_item
from training.spec_catalog import build_initial_draft_catalog, validate_catalog

import pandas as pd


def write_human_audit_summary(audit: dict) -> None:
    deep = audit.get("benchmark_v3_deep") or {}
    docs = PROJECT_ROOT / "docs/phase12"
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 12 — Dataset readiness audit",
        "",
        f"Generated: {audit.get('generated_at')}",
        "",
        "## benchmark_v3 highlights",
        "",
        f"- Checksum: `{deep.get('checksum_sha256', 'n/a')}`",
        f"- Cross-split duplicate code bodies: **{len(deep.get('cross_split_duplicate_code_hashes', []))}** (review candidates)",
        f"- Identical code / different label: **{len(deep.get('identical_code_different_label', []))}**",
        f"- Template-test pairs where fault still passes tests: **{deep.get('review_candidates_both_pass_count', 0)}**",
        f"- Families without human-reviewed NL spec: **{len(deep.get('families_without_human_reviewed_spec', []))}** / {len(deep.get('family_specification_trust', {}))}",
        "",
        "Automated findings are **review candidates**, not proof of label error.",
        "",
        "Machine-readable: `artifacts/metrics/phase12/dataset_audit.json`",
    ]
    (docs / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_eval_protocol() -> None:
    text = """# Phase 12 — Future evaluation protocol (vNext)

## Dataset versioning
- New benchmark version (e.g. `benchmark_v4`) with checksum + manifest; never overwrite v3 labels in place.
- Holdout test split **frozen** before any model selection; document `test_inspected: true/false`.

## Splits
- **Group by `family_id`** (mandatory); verify with `verify_group_isolation`.
- No template/family overlap across train/val/test.
- Report duplicate/near-duplicate code hashes **across splits** before training.

## Sample requirements
- Per-class minimum on **validation** (propose ≥5 per fault class before macro-F1 claims).
- Flag `synthetic_unreviewed`, `review_queue_excluded`, `human_adjudicated` provenance fields.

## Specifications
- Training on spec+code only when `provenance.review_status == human_reviewed`.
- Draft/machine_suggested specs may be used for analysis, not promotion gates.

## Model selection vs test
- Checkpoint / hyperparameters selected on **validation only** (single pass).
- **One** final test evaluation per experiment ID; log dataset checksum + config.
- Do not use legacy v3 test for selection (already inspected in Phases 10–11).

## Required metrics
- Macro F1, accuracy, per-class P/R/F1, confusion matrix, prediction distribution, majority-class baseline.
- Collapse guard: max single-class prediction share on val ≤ 70% for promotion consideration.

## Acceptance gate (unchanged from Phase 9)
- Val macro F1 ≥ 0.35; each fault recall ≥ 0.25; correct recall ≥ 0.50; queue templates excluded or resolved.
"""
    (PROJECT_ROOT / "docs/phase12/EVAL_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_spec_authoring_template() -> None:
    tpl = """# Problem specification authoring template (v1)

Use for **new** or **human-reviewed** entries in `data/specifications/catalog.v1.json`.

```yaml
spec_id: spec.<family_id>.v1
family_id: <family_id>
source_template_ids: [<template_ids>]
problem_statement: |
  Describe intended behavior in plain language (inputs, output, preconditions).
input_assumptions:
  - e.g. n is a non-negative integer
expected_output_summary: one line
boundary_cases:
  - empty input
  - single element
provenance:
  source: human_reviewer
  author: <name or id>
  review_status: human_reviewed  # only after real review
  created_at: ISO-8601
associated_sample_ids: []  # optional links
```

**Do not** derive `problem_statement` from `problem_type` alone or from faulty code.

## Priority families to specify first (coverage impact)
1. **palindrome** — queue + ambiguous fault taxonomy  
2. **fibonacci** — multiple queue mutations  
3. **linear_search** — index/off-by-one queue items  
4. **factorial** — baseline arithmetic family  
5. **gcd** — common beginner template (no queue flag but high utility)
"""
    (PROJECT_ROOT / "docs/phase12/SPEC_AUTHORING.md").write_text(tpl, encoding="utf-8")


def write_go_no_go(audit: dict, triage_summary: dict) -> None:
    deep = audit.get("benchmark_v3_deep") or {}
    n_spec = len(deep.get("families_without_human_reviewed_spec") or [])
    text = f"""# Phase 12 — Go / no-go for next model training

**Decision: NO-GO** for CodeBERT training experiments until:

1. At least **5 priority families** have `human_reviewed` specifications in `catalog.v1.json`.
2. Review queue items with `mechanically_verifiable` + no NL spec are adjudicated via `label_review_cli`.
3. A **new versioned** benchmark split (v4+) is cut with frozen test and zero family leakage.

Rationale: Phase 11 showed collapse with only `[TASK] problem_type`; v3 has no NL specs in CSV. Template tests pass on many fault rows — labels are not reliably behavior-distinguished.

Triage summary: {json.dumps(triage_summary, indent=2)}

This phase did **not** train models or modify production.
"""
    (PROJECT_ROOT / "docs/phase12/GO_NO_GO.md").write_text(text, encoding="utf-8")


def main() -> int:
    spec_dir = PROJECT_ROOT / "data" / "specifications"
    spec_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = spec_dir / "catalog.v1.json"
    if not catalog_path.is_file():
        cat = build_initial_draft_catalog()
        catalog_path.write_text(json.dumps(cat, indent=2), encoding="utf-8")
    ok, errs = validate_catalog(json.loads(catalog_path.read_text(encoding="utf-8")))
    if not ok:
        print("Catalog validation:", errs)

    audit = run_audit()
    out = PROJECT_ROOT / "artifacts/metrics/phase12"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv")
    triage = [triage_queue_item(df, item) for item in queue]
    packet = build_prioritized_packet(triage)
    (out / "review_triage.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")

    write_human_audit_summary(audit)
    write_eval_protocol()
    write_spec_authoring_template()
    write_go_no_go(audit, packet.get("summary", {}))

    readme = PROJECT_ROOT / "docs/phase12/README.md"
    readme.write_text(
        """# Phase 12 — Dataset readiness

```powershell
py -3.13 -m training.phase12_run
```

| Deliverable | Path |
|-------------|------|
| Dataset audit | [DATA_AUDIT.md](./DATA_AUDIT.md), `artifacts/metrics/phase12/dataset_audit.json` |
| Review triage | [REVIEW_TRIAGE.md](./REVIEW_TRIAGE.md), `artifacts/metrics/phase12/review_triage.json` |
| Spec catalog | `data/specifications/catalog.v1.json`, [SPEC_AUTHORING.md](./SPEC_AUTHORING.md) |
| Eval protocol | [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) |
| Go/no-go | [GO_NO_GO.md](./GO_NO_GO.md) |
""",
        encoding="utf-8",
    )
    print("Phase 12 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
