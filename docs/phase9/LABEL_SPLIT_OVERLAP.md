# Label distribution and split overlap

Source: `artifacts/metrics/phase9/dataset_audit_report.json` (2026-10-20).

## Legacy v1 — counts per split

| Split | n | correct | wrong_init | off_by_one | wrong_op | wrong_cond |
|-------|---|---------|------------|------------|----------|------------|
| train | 61 | 28 | 7 | 6 | 11 | 9 |
| val | 14 | 6 | 3 | 1 | 3 | 1 |
| test | 16 | 8 | 2 | 2 | 2 | 2 |

**Imbalance:** ~46% correct overall; test is 50% correct (8/16).

**Split isolation (verified):** no `source_template_id` appears in more than one split.

**Problem-type overlap (concern, by design):** same `problem_type` in multiple splits (e.g. factorial in train/val/test). Models can exploit family-level token patterns without template ID leakage.

## benchmark_v2 — counts per split

| Split | n | Families | correct | fault classes |
|-------|---|----------|---------|---------------|
| train | 74 | 12 | 38 | 36 fault |
| val | 16 | 3 | 10 | 6 fault |
| test | 31 | 5 | 18 | 13 fault |

Manifest: test families `clamp`, `factorial`, `is_sorted`, `linear_search`, `sorting`.

## benchmark_v3 — counts per split

| Split | n | Families | correct | wrong_init | off_by_one | wrong_op | wrong_cond |
|-------|---|----------|---------|------------|------------|----------|------------|
| train | 94 | 31 | 72 | 7 | 4 | 6 | 5 |
| val | 30 | 7 | 20 | 3 | 2 | 4 | 1 |
| test | 66 | 15 | 40 | 4 | 4 | 8 | 10 |

**Val fault coverage:** only **1** `wrong_condition` example in val — weak for thresholding that class.

## Duplicates & near-duplicates

| Dataset | Duplicate IDs | Duplicate normalized bodies | Near-dup scan |
|---------|---------------|----------------------------|---------------|
| legacy_v1 | 0 | 0 | capped sample in JSON |
| benchmark_v2 | 0 | 0 | see audit JSON |
| benchmark_v3 | 0 | 0 | see audit JSON |

Mutations are mostly template-local; near-duplicates across templates are limited but **correct** variants share structural tokens heavily (CodeBERT mean-pool similarity issue — see similarity audit).

## Leakage summary

| Risk | legacy_v1 | benchmark_v2 | benchmark_v3 |
|------|-----------|--------------|--------------|
| Template/family ID across splits | **None verified** | **None verified** | **None verified** |
| Same problem_type across splits | Yes (concern) | Reduced (family_id) | Reduced (family_id) |
| Test set used for dev decisions | **Yes — contaminated** | **Yes — documented** | Use only if experiments avoid test tuning |
| Label–test inconsistency | Not queue-file | v2 review artifacts | **10 verified queue items** |

## Holdout contamination (evaluation hygiene)

| Split | Status |
|-------|--------|
| legacy v1 test | Used for production `best_val_f1` / reported production metrics — **do not use for new model selection** |
| benchmark_v2 test | Marked inspected in collapse audit |
| benchmark_v3 test | Preferred **fresh** holdout **if** no experiment in repo tuned on it (verify per run logs) |
