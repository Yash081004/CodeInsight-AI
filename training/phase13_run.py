"""
Phase 13 orchestrator — QA prep docs, queue reconciliation, spec packets, benchmark v4.

Usage: py -3.13 -m training.phase13_run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.benchmark_v4_build import build_v4
from training.phase13_queue_reconciliation import build_reconciliation_report
from training.phase13_spec_packets import load_authoring_packets, merge_packets_into_catalog


def write_queue_mismatch_doc(report: dict) -> None:
    docs = PROJECT_ROOT / "docs/phase13"
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 13 — Review queue mismatch analysis",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Root cause (evidence-based)",
        "",
        "Phase 12 triage labeled **5** items `insufficient_evidence` because no CSV row matched "
        "`template_id + intended_label + mutation_type`. Reconciliation shows those queue entries "
        "refer to **generator mutations that still pass bundled tests** and were **never written** "
        "to `dataset.csv` (see `generate_benchmark.py`: `continue` after `review_queue.append`).",
        "",
        "This is **not** missing CSV data; it is **queue-only excluded mutations**.",
        "",
        "Four additional items match CSV rows but share **taxonomy collisions** (multiple mutations "
        "with the same label + mutation_type) or **stale queue reasons** vs hash-linked CSV samples.",
        "",
        "## Summary by mismatch class",
        "",
    ]
    for k, v in sorted((report.get("summary_by_mismatch_class") or {}).items()):
        lines.append(f"- **{k}**: {v}")
    lines.extend(["", "## Per-item trace", ""])
    for item in report.get("items") or []:
        q = item["queue_item"]
        lines.append(
            f"### `{q['template_id']}` / `{q['intended_label']}` / `{q.get('mutation_type')}`"
        )
        lines.append(f"- **Class:** {item['mismatch_class']}")
        lines.append(f"- **CSV strict matches:** {item.get('csv_strict_sample_ids') or 'none'}")
        for ev in item.get("evidence") or []:
            lines.append(f"- {ev}")
        if item.get("generator_matches"):
            gm = item["generator_matches"][0]
            lines.append(
                f"- Generator: pass={gm.get('bundled_tests_pass')} — {gm.get('mutation_description')}"
            )
        lines.append("")
    (docs / "QUEUE_MISMATCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_review_packet(report: dict) -> None:
    docs = PROJECT_ROOT / "docs/phase13"
    priority = [
        r for r in report.get("items") or [] if r["mismatch_class"] == "queue_only_excluded_mutation"
    ] + [
        r
        for r in report.get("items") or []
        if r["mismatch_class"]
        in ("taxonomy_collision_multiple_mutations", "stale_queue_reason_vs_csv_sample")
    ]
    lines = [
        "# Phase 13 — Proposed review packet (no decisions recorded)",
        "",
        "Grouped to minimize reviewer effort. **human_reviewed: false** for all items.",
        "",
        "## Batch A — Queue-only mutations (not in CSV)",
        "",
        "Reviewer needs: confirm whether mutation should enter a future benchmark with stronger tests, "
        "be relabeled, or remain excluded.",
        "",
    ]
    for item in priority:
        if item["mismatch_class"] != "queue_only_excluded_mutation":
            continue
        q = item["queue_item"]
        lines.append(f"- **{q['template_id']}** `{q['intended_label']}` ({q.get('mutation_type')})")
        for gm in item.get("generator_matches") or []:
            if gm.get("bundled_tests_pass"):
                lines.append(f"  - Passing mutation: {gm.get('mutation_description')}")
    lines.extend(
        [
            "",
            "## Batch B — CSV linked but queue reason or taxonomy ambiguous",
            "",
        ]
    )
    for item in priority:
        if item["mismatch_class"] == "queue_only_excluded_mutation":
            continue
        q = item["queue_item"]
        lines.append(
            f"- **{q['template_id']}** `{q['intended_label']}` — samples "
            f"{item.get('csv_strict_sample_ids')} — _{item['mismatch_class']}_"
        )
    (docs / "REVIEW_PACKET.md").write_text("\n".join(lines), encoding="utf-8")


def write_spec_status_doc() -> None:
    packets = load_authoring_packets()
    docs = PROJECT_ROOT / "docs/phase13"
    lines = [
        "# Phase 13 — Specification authoring packets",
        "",
        "All packets are **draft** (`review_status: draft`, `human_reviewed: false`).",
        "",
        "| Family | Packet file | Catalog spec_id |",
        "|--------|-------------|-----------------|",
    ]
    for p in packets:
        fam = p["family_id"]
        path = f"data/specifications/packets/{fam}.authoring_packet.json"
        lines.append(f"| {fam} | `{path}` | `spec.{fam}.phase13_draft` |")
    lines.extend(
        [
            "",
            "Human reviewers must approve before setting `review_status: human_reviewed` in "
            "`data/specifications/catalog.v1.json`.",
        ]
    )
    (docs / "SPEC_PACKETS.md").write_text("\n".join(lines), encoding="utf-8")


def write_v4_readiness(manifest: dict) -> None:
    docs = PROJECT_ROOT / "docs/phase13"
    text = f"""# Phase 13 — Benchmark v4 readiness

## Status

- **Infrastructure:** `training/benchmark_v4_build.py`, `data/benchmark_v4/manifest.json`
- **Eligible rows (current catalog):** {manifest.get("eligible_row_count")} / 190 v3 rows
- **Quarantined:** {manifest.get("quarantine_row_count")}
- **Test split frozen:** `data/benchmark_v4/classification/test_split_frozen.json`
- **Pristine test claim:** **No** (`pristine_claim: false` until human specs + queue adjudication)

## Quarantine reasons (counts)

```json
{json.dumps(manifest.get("quarantine_reason_counts"), indent=2)}
```

## v4 readiness for training

**Not ready** — zero eligible rows until at least priority families have `human_reviewed` specifications and review-queue templates are resolved or excluded by policy.

v3 files and labels are unchanged.
"""
    (docs / "BENCHMARK_V4_READINESS.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    (PROJECT_ROOT / "docs/phase13/README.md").write_text(
        """# Phase 13 — Restore QA and prepare reviewed benchmark

```powershell
py -3.13 -m training.phase13_run
py -3.13 scripts/run_qa.py
```

| Deliverable | Path |
|-------------|------|
| Queue mismatch | [QUEUE_MISMATCH_REPORT.md](./QUEUE_MISMATCH_REPORT.md) |
| Review packet | [REVIEW_PACKET.md](./REVIEW_PACKET.md) |
| Spec packets | [SPEC_PACKETS.md](./SPEC_PACKETS.md) |
| Benchmark v4 | [BENCHMARK_V4_READINESS.md](./BENCHMARK_V4_READINESS.md) |
| Metrics | `artifacts/metrics/phase13/` |
""",
        encoding="utf-8",
    )


def main() -> int:
    report = build_reconciliation_report()
    out = PROJECT_ROOT / "artifacts/metrics/phase13"
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    merge_packets_into_catalog(write=True)
    manifest = build_v4(write_csv=True)

    write_queue_mismatch_doc(report)
    write_review_packet(report)
    write_spec_status_doc()
    write_v4_readiness(manifest)
    write_readme()
    print("Phase 13 artifacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
