"""
Phase 11 — E3 controlled comparison: code-only vs spec+code on filtered benchmark v3.

Usage:
  py -3.13 -m training.phase11_experiments
  py -3.13 -m training.phase11_experiments --control-only
  py -3.13 -m training.phase11_experiments --spec-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_validation import sha256_file
from training.phase10_experiments import GATE, evaluate_acceptance_gate
from training.phase10_v3_exclusions import verify_no_family_leakage
from training.spec_code_input import audit_specifications, truncate_strategy_doc

FILTERED_VIEW = PROJECT_ROOT / "artifacts/metrics/phase10/E1_filtered_dataset_view.csv"
EXCLUSION_MANIFEST = PROJECT_ROOT / "artifacts/metrics/phase10/E1_codebert_v3/exclusion_manifest.json"
PHASE11 = PROJECT_ROOT / "artifacts/metrics/phase11"
CONTROL_METRICS = PHASE11 / "E3_control"
SPEC_METRICS = PHASE11 / "E3_spec_code"
CONTROL_MODEL = PROJECT_ROOT / "models/experiments/phase11_e3_control"
SPEC_MODEL = PROJECT_ROOT / "models/experiments/phase11_e3_spec_code"

TRAIN_KW = dict(
    epochs=3,
    batch_size=4,
    eval_batch_size=8,
    lr=2e-5,
    max_length=256,
    seed=42,
    skip_preflight=True,
    no_class_weights=False,
    balanced_sampler=False,
    patience=0,
)


def _load_eligibility() -> Dict[str, Any]:
    if not FILTERED_VIEW.is_file():
        raise FileNotFoundError(f"Missing filtered view: {FILTERED_VIEW}. Run Phase 10 E1 first.")
    df = pd.read_csv(FILTERED_VIEW)
    checksum = sha256_file(FILTERED_VIEW)
    ok, errs = verify_no_family_leakage(df)
    spec_audit = audit_specifications(df)
    manifest = json.loads(EXCLUSION_MANIFEST.read_text()) if EXCLUSION_MANIFEST.is_file() else {}
    splits = df.groupby("split").size().to_dict()
    return {
        "filtered_view_path": str(FILTERED_VIEW.relative_to(PROJECT_ROOT)),
        "filtered_view_checksum_sha256": checksum,
        "split_counts": splits,
        "sample_ids_by_split": {
            sp: df[df["split"] == sp]["sample_id"].tolist() for sp in sorted(df["split"].unique())
        },
        "spec_audit": spec_audit,
        "exclusion_manifest_ref": str(EXCLUSION_MANIFEST.relative_to(PROJECT_ROOT)),
        "excluded_sample_ids": manifest.get("excluded_sample_ids", []),
        "family_leakage_ok": ok,
        "family_leakage_errors": errs[:5] if errs else [],
    }


def _run_train(condition: str, input_mode: str, metrics_dir: Path, model_dir: Path) -> Dict[str, Any]:
    from training.train_codebert_classifier import train

    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    class Args:
        pass

    args_obj = Args()
    args_obj.data_path = str(FILTERED_VIEW)
    args_obj.output_dir = str(model_dir)
    args_obj.metrics_dir = str(metrics_dir)
    args_obj.model_name = "microsoft/codebert-base"
    args_obj.experiment_id = f"phase11_{condition}"
    args_obj.input_mode = input_mode
    args_obj.skip_test_eval = False
    for k, v in TRAIN_KW.items():
        setattr(args_obj, k, v)

    train(args_obj)
    runtime = round(time.time() - t0, 2)
    val_m = json.loads((metrics_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    test_m = {}
    if (metrics_dir / "codebert_classification_metrics.json").is_file():
        test_m = json.loads((metrics_dir / "codebert_classification_metrics.json").read_text(encoding="utf-8"))
    total = sum(val_m.get("predicted_label_counts", {}).values()) or 1
    val_m["max_predicted_class_fraction"] = max(val_m.get("predicted_label_counts", {0: 0}).values()) / total
    gate = evaluate_acceptance_gate(
        {
            "macro_f1": val_m.get("macro_f1", 0),
            "per_class_metrics": val_m.get("per_class_metrics", {}),
            "max_predicted_class_fraction": val_m.get("max_predicted_class_fraction", 1),
        },
        queue_excluded_from_train=True,
    )
    summary = {
        "condition": condition,
        "input_mode": input_mode,
        "runtime_seconds": runtime,
        "validation": val_m,
        "test": test_m,
        "acceptance_gate": gate,
        "training_config": json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
        if (model_dir / "training_config.json").is_file()
        else {},
    }
    (metrics_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_review_packet() -> Path:
    queue = json.loads(
        (PROJECT_ROOT / "data/benchmark_v3/classification/review_queue.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Phase 11 — Review queue packet (not human-reviewed)",
        "",
        "These items are **verified** `mutation_still_passes_tests` flags. "
        "No label changes without `training.label_review_cli decide`.",
        "",
        "| Priority | template_id | intended_label | mechanical check | human needed |",
        "|----------|-------------|----------------|------------------|--------------|",
    ]
    for i, item in enumerate(queue, 1):
        tid = item.get("template_id")
        mechanical = "Bundled generator tests still pass"
        human = "Confirm fault taxonomy vs intended_label"
        pri = "high" if tid in ("fibonacci_1", "palindrome_1", "linear_search_2") else "medium"
        lines.append(
            f"| {pri} | {tid} | {item.get('intended_label')} | {mechanical} | {human} |"
        )
    lines.extend(
        [
            "",
            "CLI: `py -3.13 -m training.label_review_cli --dataset benchmark_v3 list`",
            "Packet: `py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027`",
        ]
    )
    path = PROJECT_ROOT / "docs/phase11/REVIEW_PACKET.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_comparison(control: Dict, spec: Dict, eligibility: Dict) -> None:
    docs = PROJECT_ROOT / "docs/phase11"
    docs.mkdir(parents=True, exist_ok=True)
    hist_e1_val = 0.2105
    hist_e0_val = 0.3267
    c_val = control["validation"].get("macro_f1", 0)
    s_val = spec["validation"].get("macro_f1", 0)
    improved = s_val > c_val + 0.01
    body = f"""# Phase 11 — E3 spec+code experiment

Generated: {datetime.now(timezone.utc).isoformat()}

## Data eligibility

- Filtered view checksum: `{eligibility['filtered_view_checksum_sha256']}`
- Splits: {eligibility['split_counts']}
- Excluded (train/val only): {len(eligibility.get('excluded_sample_ids', []))} samples per Phase 10 manifest
- Family leakage after filter: **{'OK' if eligibility['family_leakage_ok'] else 'FAIL'}**

### Specification audit

- Natural-language problem statements in CSV: **none**
- `expected_behavior`: only `{eligibility['spec_audit'].get('expected_behavior_unique')}` (test outcomes, not specs)
- E3 `spec_code` uses factual `[TASK] problem_type` + `[CODE]` — see `training/spec_code_input.py`

Truncation: {truncate_strategy_doc(256)}

## Controlled comparison (same eligible rows, seed 42, 3×256)

| Condition | Val macro F1 | Gate | Max class share (val) |
|-----------|--------------|------|------------------------|
| Control (code-only) | **{c_val}** | {'PASS' if control['acceptance_gate']['passed'] else 'FAIL'} | {control['validation'].get('max_predicted_class_fraction', 'n/a')} |
| Spec+code (task prefix) | **{s_val}** | {'PASS' if spec['acceptance_gate']['passed'] else 'FAIL'} | {spec['validation'].get('max_predicted_class_fraction', 'n/a')} |

Historical (different run id): Phase 10 E1 val **{hist_e1_val}**, E0 TF-IDF val **{hist_e0_val}**.

Spec input materially improved over control: **{'yes' if improved else 'no'}** (threshold +0.01 macro F1).

## Test metrics (single eval each; v3 test previously inspected in Phase 10)

- Control test macro F1: {control.get('test', {}).get('macro_f1', 'n/a')}
- Spec test macro F1: {spec.get('test', {}).get('macro_f1', 'n/a')}

## Recommendation

{"Investigate **data quality / real problem statements** before more CodeBERT training; task slug alone is insufficient spec." if not improved else "Consider E3 follow-up with real reviewed specs — task prefix shows signal."}

Do **not** promote to production.
"""
    (docs / "RESULTS.md").write_text(body, encoding="utf-8")
    (PHASE11 / "comparison.json").write_text(
        json.dumps(
            {
                "eligibility": eligibility,
                "control": control,
                "spec_code": spec,
                "historical_e0_val_macro_f1": hist_e0_val,
                "historical_e1_val_macro_f1": hist_e1_val,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--spec-only", action="store_true")
    args = parser.parse_args()
    run_control = not args.spec_only
    run_spec = not args.control_only

    eligibility = _load_eligibility()
    PHASE11.mkdir(parents=True, exist_ok=True)
    (PHASE11 / "eligibility.json").write_text(json.dumps(eligibility, indent=2), encoding="utf-8")
    write_review_packet()

    control_summary: Optional[Dict] = None
    spec_summary: Optional[Dict] = None
    if run_control:
        print("[Phase 11] Control (code-only)...")
        control_summary = _run_train("E3_control", "code_only", CONTROL_METRICS, CONTROL_MODEL)
        print(f"  val macro F1={control_summary['validation'].get('macro_f1')}")
    if run_spec:
        print("[Phase 11] Experiment (spec+code)...")
        spec_summary = _run_train("E3_spec_code", "spec_code", SPEC_METRICS, SPEC_MODEL)
        print(f"  val macro F1={spec_summary['validation'].get('macro_f1')}")

    if control_summary and spec_summary:
        write_comparison(control_summary, spec_summary, eligibility)
        print(f"Wrote {PROJECT_ROOT / 'docs/phase11/RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
