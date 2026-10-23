# Phase 13 — Benchmark v4 readiness

## Status

- **Infrastructure:** `training/benchmark_v4_build.py`, `data/benchmark_v4/manifest.json`
- **Eligible rows (current catalog):** 0 / 190 v3 rows
- **Quarantined:** 190
- **Test split frozen:** `data/benchmark_v4/classification/test_split_frozen.json`
- **Pristine test claim:** **No** (`pristine_claim: false` until human specs + queue adjudication)

## Quarantine reasons (counts)

```json
{
  "family_spec_not_human_reviewed": 190,
  "correct_baseline_requires_spec_context": 132,
  "review_queue_template": 35,
  "synthetic_style_variant": 66
}
```

## v4 readiness for training

**Not ready** — zero eligible rows until at least priority families have `human_reviewed` specifications and review-queue templates are resolved or excluded by policy.

v3 files and labels are unchanged.
