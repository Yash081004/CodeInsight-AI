# Phase 14 — Benchmark v4 eligibility reassessment

Strict v4 eligible: **0** / 190
Narrow (except spec): **0** rows would remain blocked by missing human-reviewed spec.

## Independent quarantine reason counts

```json
{
  "correct_baseline_requires_spec_context": 132,
  "family_spec_not_human_reviewed": 190,
  "sample_label_review_pending": 30,
  "review_queue_template": 35,
  "synthetic_style_variant": 66
}
```

## Protocol

- **Strict:** spec + no review-queue template + no style variant + correct-row policy + no pending sample review.
- **Narrow (documented only):** fault rows with `fails_tests` and failing bundled tests — still **not** promoted without human-reviewed spec.
- **Spec ≠ label:** approved specification does not clear queue-template or pending V3RQ label review.

Machine-readable: `artifacts/metrics/phase14/v4_eligibility_assessment.json`