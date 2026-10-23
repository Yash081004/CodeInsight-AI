# Phase 7 — Error analysis, label validation & trustworthy evaluation

**Generated:** 2026-10-19  
**Dataset:** `benchmark_v3` (190 samples, synthetic, **not human-reviewed**)  
**Checksum:** `9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383`  
**Research model analyzed:** CodeBERT `gpu_3_balanced` (not production)  
**Artifacts:** `artifacts/metrics/phase7/error_analysis_report.json`, `error_explorer_entries.json`

Regenerate analysis (no training):

```bash
py -3.13 -m training.phase7_error_analysis
```

---

## 1. Verified facts (evidence-backed)

| Finding | Evidence | Example IDs |
|---------|----------|-------------|
| CodeBERT val softmax often nearly flat | 9 val rows with probability std &lt; 0.015 | S0025, S0026, S0027, S0028, S0106, S0110, S0119, S0173, S0185 |
| `factorial` val family: CodeBERT over-predicts `wrong_initialization` | 6/13 val preds `wrong_initialization`; true mix includes off-by-one & operator faults | S0001–S0006 (subset) |
| Val zero recall on `off_by_one_loop`, `wrong_condition` for CodeBERT | `validation_metrics.json` per-class recall 0.0 (support 2 and 1) | S0003 (off_by_one), S0005-area |
| TF-IDF val collapse to `correct` | 25/30 preds `correct`; macro F1 26.89% | `runner/tfidf_val_metrics.json` |
| Both models wrong on same val IDs | Independent val replay | **S0003, S0009, S0027, S0028** |

**Interpretation limits:** Flat softmax shows **weak class separation**, not proof of lexical-only shortcuts. Template/family token correlation with TF-IDF was **not measured** in this phase.

---

## 2. Hypotheses (not proven)

- **Balanced sampler → `wrong_initialization` bias:** Consistent with 14/30 val preds as `wrong_initialization` and training oversampling of minority faults (`gpu_3_balanced/validation_metrics.json`). Causal link **not established**.

---

## 3. Validation metrics summary

| Model | Macro F1 (val) | Collapse (val) | Notes |
|-------|----------------|----------------|-------|
| TF-IDF (train-fit replay) | 0.2689 | Yes (~83% `correct`) | `phase7/tfidf_val_predictions.json` |
| CodeBERT `gpu_3_balanced` | 0.3459 | No | Research checkpoint only |

**Test (historical, post Phase 6 — not used for tuning in Phase 7):** CodeBERT test macro F1 **0.2713** (`final_test_metrics.json`).

---

## 4. Task contract (classification API)

The default product path classifies **source code only** (`POST /api/classification/analyze`).

| Fault type | Inferable from code alone? | Typically needs |
|------------|----------------------------|-----------------|
| `wrong_initialization` | Often | Problem statement for identity choice |
| `off_by_one_loop` | Partial | Intended inclusive/exclusive bounds |
| `wrong_operator` | Partial | Intended min/max or comparison semantics |
| `wrong_condition` | Partial | Intended predicate / I/O spec |
| `correct` | Partial | Full behavioral specification |

**Backward-compatible extensions (optional request fields):**

- `problem_statement` — intended behavior  
- `function_context` — role / language context  
- `test_cases` + `run_tests` — tests are **not executed** unless sandbox is enabled; UI/API state this explicitly  

Implementation: `backend/app/schemas.py`, `training/task_context.py`, `explainability_service.py`.

---

## 5. Human review workflow

**Queue:** `data/benchmark_v3/review_decisions.json` (40 prioritized items, all **`pending`**)  
**CLI:**

```bash
py -3.13 -m training.label_review_cli --dataset benchmark_v3 list
py -3.13 -m training.label_review_cli --dataset benchmark_v3 show --id V3RQ001
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --reviewer "name"
```

Original `dataset.csv` labels are **not** overwritten; decisions append `audit_history`.

**Reviewed eval slice:** `data/benchmark_reviewed/manifest.json` → **`blocked_pending_human_review`**. Export when adjudication exists:

```bash
py -3.13 -m training.build_reviewed_eval_slice
```

---

## 6. Evaluation Lab

- `GET /api/evaluation/phase7-error-analysis` — report summary  
- `GET /api/evaluation/error-explorer` — filters: split, model (`codebert`|`tfidf`), labels, family_id (validated)  
- `GET /api/evaluation/review-queue` — review status & slice blocker  

UI: Evaluation Lab → Phase 7 error explorer panel.

---

## 7. Production & benchmark preservation

- **Production:** `models/codebert_classifier/` unchanged  
- **benchmark_v2** checksum unchanged  
- **benchmark_v3** checksum unchanged  
- Phase 6 metrics and checkpoints preserved under `artifacts/metrics/benchmark_v3/` and `models/benchmark_v3/`

---

## 8. Recommended next steps (evidence-based)

1. Complete human adjudication for priority queue (`off_by_one_loop`, `wrong_condition`, dual-model errors).  
2. Export reviewed slice only after **accepted/relabeled** decisions with rationale.  
3. Revisit generator templates where `validation_status` / finite tests disagree with intended fault type.  
4. Do **not** tune on v3 test; any new training requires a fresh benchmark version after label fixes.  
5. Optional GPU re-run of Phase 6 matrix only after label quality improves.
