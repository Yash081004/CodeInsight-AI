# Phase 11 — E3 spec+code experiment

Generated: 2026-10-20T06:48:59.402789+00:00

## Data eligibility

- Filtered view checksum: `39b21118a1d86e6b2d2cd8e4f4c64551351f2de1598067c0e57aeee314013090`
- Splits: {'test': 66, 'train': 84, 'val': 22}
- Excluded (train/val only): 18 samples per Phase 10 manifest
- Family leakage after filter: **OK**

### Specification audit

- Natural-language problem statements in CSV: **none**
- `expected_behavior`: only `['fails_tests', 'passes_all_tests']` (test outcomes, not specs)
- E3 `spec_code` uses factual `[TASK] problem_type` + `[CODE]` — see `training/spec_code_input.py`

Truncation: Single sequence tokenized with max_length=256, truncation=True (tail truncated). Format: [TASK] problem_type [SEP] [CODE] source. Spec prefix is preserved when possible; long code may be truncated from the end.

## Controlled comparison (same eligible rows, seed 42, 3×256)

| Condition | Val macro F1 | Gate | Max class share (val) |
|-----------|--------------|------|------------------------|
| Control (code-only) | **0.2105** | FAIL | 1.0 |
| Spec+code (task prefix) | **0.2105** | FAIL | 1.0 |

Historical (different run id): Phase 10 E1 val **0.2105**, E0 TF-IDF val **0.3267**.

Spec input materially improved over control: **no** (threshold +0.01 macro F1).

## Test metrics (single eval each; v3 test previously inspected in Phase 10)

- Control test macro F1: 0.1509
- Spec test macro F1: 0.1509

## Recommendation

Investigate **data quality / real problem statements** before more CodeBERT training; task slug alone is insufficient spec.

Do **not** promote to production.
