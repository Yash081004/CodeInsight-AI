# Phase 10 — E0/E1 validation comparison & acceptance gate

Generated: 2026-10-20 (after E0 + E1 runs)

Dataset checksum (full v3): `9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383`

## E0 TF-IDF (benchmark v3, full train/val/test)

- **Hyperparameters:** `class_weight=balanced`, **C selected on val** from `[0.01, 0.1, 1, 10]` → **C=10.0** (val macro F1 0.3267).
- **Val:** macro F1 **0.3267**, accuracy 0.733, max predicted class share **93.3%** (`correct` + partial `wrong_initialization`).
- **Test (single eval after selection):** macro F1 **0.302**, accuracy 0.652.
- **Fault recall on val:** wrong_init 0.67; off_by_one, wrong_operator, wrong_condition **0**.
- **Gate passed:** **NO** — macro F1 &lt; 0.35; max class fraction &gt; 0.70; three fault recalls &lt; 0.25.

## E1 CodeBERT (review-queue templates excluded from train/val)

- **Config:** 3 epochs, max_length **256**, batch 4, lr 2e-5, seed 42, weighted CE, **~295 s CPU**.
- **Exclusions:** 18 rows, 8 `source_template_id`s from review queue; **test split unchanged** (66 samples). Filtered view checksum `39b21118…13090`.
- **Train/val after filter:** 84 train, 22 val (106 train+val rows).
- **Val (checkpoint selection):** macro F1 **0.2105**, **100%** of predictions `correct` (collapse).
- **Test (one final eval, best epoch 1):** macro F1 **0.1509**, accuracy 0.606 — **0 fault recall** all classes.
- **Gate passed:** **NO** — all fault recalls 0 on val; macro F1 far below 0.35; collapse.

## Comparison on validation

| Metric | E0 | E1 |
|--------|----|----|
| Macro F1 | **0.327** | 0.211 |
| Max class pred share | 0.93 | **1.00** |
| Collapse to `correct` | partial | **full** |

**E0 outperforms E1** on v3 validation; CodeBERT did not beat TF-IDF.

## Acceptance thresholds (unchanged)

- Macro F1 ≥ **0.35**
- Each fault recall ≥ **0.25**
- `correct` recall ≥ **0.50**
- Max single-class prediction share ≤ **0.70**
- Review-queue templates excluded from E1 train/val (**done**)

## Promotion

**Do not promote** either candidate to production.

## Test-set caveat

v3 test was used **once per experiment** for reporting after val-based selection (E0: C; E1: epoch). It was **not** used to pick between E0 and E1. Prior project work may have inspected v3 test families — do not treat as a fresh global holdout across all historical runs.

## Next experiment

1. **E3:** Train CodeBERT on `[SPEC] expected_behavior + [CODE] code` using the **same E1 filtered view** (3 epochs, max_len 256) — tests whether under-specified code-only input is the bottleneck.
2. **In parallel (data):** Human adjudication of the 10 review-queue templates; do **not** change CSV labels until decisions are recorded in review workflow artifacts.
