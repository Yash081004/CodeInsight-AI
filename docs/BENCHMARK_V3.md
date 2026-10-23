# benchmark_v3

Versioned dataset under `data/benchmark_v3/`. **Does not replace** `data/benchmark_v2/`.

## Generate

```powershell
py -3.13 -m training.dataset_pipeline.generate_benchmark_v3
py -3.13 -m training.phase5_audit
py -3.13 -m training.benchmark_runner --dataset benchmark_v3 --split val
py -3.13 -m training.gpu_experiment_readiness
```

## Design

- All v1/v2 templates plus **32 new program families** (`benchmark_v3_templates.py`).
- Family-grouped splits with **label-coverage refinement** (`splits_v3.py`).
- `human_reviewed: false`, `synthetic_unreviewed: true` in quality report.
- Behavioral tests are finite; review queue items stay out of accepted CSV until human review (`training/label_review_cli.py`).

## Review workflow

```powershell
py -3.13 -m training.label_review_cli list
py -3.13 -m training.label_review_cli decide --id RQ001 --status rejected --rationale "..." --reviewer name
```

Decisions: `data/benchmark_v2/review_decisions.json` (pending until a human decides).

## GPU training

See `artifacts/metrics/benchmark_v3/gpu_readiness.json` — matrix proposed, **not auto-run**.
