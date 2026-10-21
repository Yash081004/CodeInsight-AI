# Phase 12 — Future evaluation protocol (vNext)

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
