# Phase 9 — Dataset audit and model improvement plan

Read-only audit completed **2026-10-20**. No benchmark labels, checkpoints, or production weights were modified.

| Deliverable | Path |
|-------------|------|
| Dataset audit (narrative) | [DATASET_AUDIT.md](./DATASET_AUDIT.md) |
| Label distribution & split overlap | [LABEL_SPLIT_OVERLAP.md](./LABEL_SPLIT_OVERLAP.md) |
| Model failure analysis | [MODEL_FAILURE_ANALYSIS.md](./MODEL_FAILURE_ANALYSIS.md) |
| Experiment matrix & acceptance criteria | [EXPERIMENT_MATRIX.md](./EXPERIMENT_MATRIX.md) |
| Machine-readable audit | [`artifacts/metrics/phase9/dataset_audit_report.json`](../../artifacts/metrics/phase9/dataset_audit_report.json) |

Regenerate audit JSON:

```powershell
py -3.13 -m training.phase9_dataset_audit
```
