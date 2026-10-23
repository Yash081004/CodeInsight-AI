# Dataset audit — classification & similarity

## Scope

Three classification corpora and legacy similarity pairs were inspected via `training/phase9_dataset_audit.py`, `training/validate_dataset`, generator code (`training/generate_dataset.py`, `training/dataset_pipeline/*`), and existing validation JSON. **Labels were not edited.**

## Classification corpora

### Legacy v1 (production training path today)

| Field | Value |
|-------|--------|
| Path | `data/classification/dataset.csv` |
| Checksum | `c7ae7a8d836b64c9aa139e9c1fe77724329840a5629f87a08b3a2d1e993aff37` |
| Samples | 91 |
| Split units | `source_template_id` (grouped, seed 42) |
| Template IDs | 21 |
| Problem-type families | 12 distinct `problem_type` values |

**Per-label totals:** correct 42, wrong_operator 16, wrong_initialization 12, wrong_condition 12, off_by_one_loop 9.

**Verified (automated):** no duplicate sample IDs; no duplicate normalized code bodies; all labels ∈ five-class schema; all code AST-valid; no `validation_status` ≠ valid; **no template-level split leakage** (each `source_template_id` in one split only).

**Concerns (not auto-verified as label errors):**

- **Small n:** test has only **16** samples (2 per fault class). Metrics are high-variance; a single collapse dominates macro F1.
- **Same problem_type across splits:** e.g. `factorial`, `fibonacci`, `palindrome` appear in train/val/test under *different* template IDs. This is intentional grouped splitting but means ** lexical/style features of a problem family can generalize across splits** even without template ID leakage.
- **Synthetic ground truth:** mutations are defined by edit rules + bundled asserts in `generate_dataset.py`. Human review has **not** validated every label against a written spec.
- **`expected_behavior`:** present in CSV; not fed to the production classifier (code-only).

### benchmark_v2

| Field | Value |
|-------|--------|
| Checksum | `6d2d6ab93918ef86ebbb36d46b0ced5f109dbc0983f2229a067fe72ef194ba55` |
| Samples | 121 |
| Independent `family_id` | 20 (manifest) |
| Split | family-level holdout (5 test families) |

Stronger than v1 but still synthetic. **Test split marked contaminated** in `artifacts/metrics/benchmark_v2/phase4/collapse_audit.json` (“previously inspected”).

### benchmark_v3 (recommended forward benchmark)

| Field | Value |
|-------|--------|
| Checksum | `9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383` |
| Accepted samples | 190 |
| Independent `family_id` | 53 |
| Rejected generation | 4 |
| **`human_reviewed`** | **false** (`data/benchmark_v3/quality_report.json`) |

**Review queue (verified):** 10 template/label pairs flagged `mutation_still_passes_tests` in `data/benchmark_v3/classification/review_queue.json` — the mutation **does not fail** the generator’s bundled tests, so the assigned fault class is **not reliably observable** from behavior alone. These are **verified label–test inconsistencies**, not guesses.

Examples (template → intended label → reason):

- `fibonacci_1` → wrong_operator / wrong_condition — mutation still passes tests  
- `palindrome_1` → off_by_one_loop — mutation still passes tests  

**Ambiguous / multi-category risk (concern):** edit rules can produce code where a human would describe multiple bugs (e.g. boundary vs off-by-one). The pipeline forces **exactly one** `error_label` per row.

## Similarity data

| Corpus | Test pairs (evaluable) | Notes |
|--------|------------------------|--------|
| Legacy `data/similarity/pairs.csv` | **7** in test split | 91 cross-split pairs excluded from eval |
| benchmark_v2/v3 | larger | used in training scripts |

Reported hybrid policy metrics (`artifacts/metrics/similarity_hybrid_policy.json`) are on **n=7** test pairs — useful for regression/diagnostics, **not** for claiming production semantic similarity quality.

## Invalid Python & label mapping

- **Verified:** 0 syntax-invalid rows in all three classification CSVs under AST check in phase9 audit.
- **Verified:** 0 invalid label strings.
- **Production API** uses the same five labels as training (`backend/app/config.py`).

## Spec sufficiency

- Generator attaches `expected_behavior` text and optional tests for **generation QA**, not for model inputs.
- For **inference**, spec-assisted routing uses a **small rule set** (`spec_hint_service`) — not the dataset’s full `expected_behavior` field.
- Many v3 review-queue items show that **tests alone are insufficient** to justify the synthetic label; problem statements would need human review to defend fault class.

## Similarity vs classification

Do not tune similarity thresholds on the 7-pair legacy test set for “accuracy” claims; policy was grid-searched on val with documented methodology in `similarity_hybrid_policy.json`.
