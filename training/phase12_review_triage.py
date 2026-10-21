"""
Phase 12: triage benchmark_v3 review queue (no human decisions).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.phase12_template_tests import run_template_tests_on_code
from training.review_workflow import build_review_packet, load_decisions, template_spec

QUEUE_PATH = PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json"
V3_CSV = PROJECT_ROOT / "data/benchmark_v3/classification/dataset.csv"

NEXT_ACTION = (
    "mechanically_verifiable",
    "requires_reviewer_intended_behavior",
    "requires_expert_decision",
    "insufficient_evidence",
)


def _match_samples(df: pd.DataFrame, template_id: str, intended_label: str, mutation_type: str) -> pd.DataFrame:
    m = (df["source_template_id"] == template_id) & (df["error_label"] == intended_label)
    if mutation_type and "mutation_type" in df.columns:
        m = m & (df["mutation_type"] == mutation_type)
    return df[m]


def triage_queue_item(df: pd.DataFrame, item: Dict[str, Any]) -> Dict[str, Any]:
    tid = item["template_id"]
    label = item["intended_label"]
    mut = item.get("mutation_type", "")
    samples = _match_samples(df, tid, label, mut)
    spec = template_spec(tid)
    has_nl_spec = bool(spec.get("problem_statement"))

    mechanical: List[Dict] = []
    for _, row in samples.iterrows():
        mechanical.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "template_test_run": run_template_tests_on_code(tid, row["code"]),
            }
        )

    all_pass = all(m["template_test_run"].get("passed_all") for m in mechanical if m["template_test_run"].get("executed"))
    any_executed = any(m["template_test_run"].get("executed") for m in mechanical)

    if not len(samples):
        action = "insufficient_evidence"
        rationale = "No CSV rows match template_id + intended_label (+ mutation_type)."
    elif not any_executed:
        action = "requires_expert_decision"
        rationale = "No registry template tests available for mechanical re-check."
    elif all_pass and item.get("reason") == "mutation_still_passes_tests":
        action = "mechanically_verifiable"
        rationale = (
            "Bundled generator tests pass on fault sample(s) — confirms queue flag; "
            "human must decide if label/taxonomy still valid vs intended behavior."
        )
    elif not all_pass:
        action = "requires_reviewer_intended_behavior"
        rationale = "Template tests fail on fault sample — queue reason may be stale; reviewer compare to spec."
    else:
        action = "requires_reviewer_intended_behavior"
        rationale = "Mixed or partial test outcomes — inspect samples."

    if not has_nl_spec and action == "mechanically_verifiable":
        action = "requires_reviewer_intended_behavior"
        rationale += " No trustworthy NL spec in registry for this template."

    related_decisions = []
    for d in load_decisions("benchmark_v3").get("decisions") or []:
        if d.get("source_template_id") == tid and d.get("status") == "pending":
            related_decisions.append(d.get("review_id"))

    return {
        "queue_item": item,
        "matched_sample_ids": samples["sample_id"].tolist(),
        "matched_count": int(len(samples)),
        "has_registry_problem_statement": has_nl_spec,
        "mechanical_checks": mechanical,
        "recommended_next_action": action,
        "rationale": rationale,
        "evidence_needed_for_human": [
            "Intended behavior in natural language (not problem_type slug)",
            "Whether mutation matches assigned fault class",
            "Whether finite tests are adequate or label should be rejected/relabeled",
        ],
        "related_pending_review_ids": related_decisions,
        "human_reviewed": False,
    }


def build_prioritized_packet(triage_rows: List[Dict]) -> Dict[str, Any]:
    priority_order = {"requires_expert_decision": 0, "requires_reviewer_intended_behavior": 1, "mechanically_verifiable": 2, "insufficient_evidence": 3}
    sorted_rows = sorted(triage_rows, key=lambda r: (priority_order.get(r["recommended_next_action"], 9), -r["matched_count"]))
    groups: Dict[str, List] = defaultdict(list)
    for r in sorted_rows:
        groups[r["queue_item"]["template_id"]].append(r)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_reviewed": False,
        "grouped_by_template": dict(groups),
        "items": sorted_rows,
        "summary": {
            action: sum(1 for r in triage_rows if r["recommended_next_action"] == action) for action in NEXT_ACTION
        },
    }


def main() -> int:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(V3_CSV)
    triage = [triage_queue_item(df, item) for item in queue]
    packet = build_prioritized_packet(triage)

    out = PROJECT_ROOT / "artifacts/metrics/phase12"
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_triage.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")

    docs = PROJECT_ROOT / "docs/phase12"
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 12 — Review queue triage (no decisions recorded)",
        "",
        f"Generated: {packet['generated_at']}",
        "",
        "## Summary by next action",
        "",
    ]
    for k, v in packet["summary"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Prioritized items")
    lines.append("")
    for r in packet["items"]:
        q = r["queue_item"]
        lines.append(f"### {q['template_id']} → `{q['intended_label']}` ({q.get('mutation_type')})")
        lines.append(f"- **Next action:** {r['recommended_next_action']}")
        lines.append(f"- **Samples:** {', '.join(r['matched_sample_ids']) or 'none'}")
        lines.append(f"- **Rationale:** {r['rationale']}")
        lines.append("")
    (docs / "REVIEW_TRIAGE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'review_triage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
