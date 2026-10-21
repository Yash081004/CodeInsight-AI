# CodeInsight AI — Project Status

**Last updated:** 2026-10-19 (Phase 8 human adjudication workflow)  
**Backend Python:** 3.13.3 (`py -3.13`). Python 3.14 cannot install pinned Pydantic/FastAPI on this machine.

---

## 1. Dataset validation (2026-10-19)

**Tooling:** `py -3.13 -m training.validate_dataset`  
**Reports:** `artifacts/metrics/classification_dataset_validation.json`, `similarity_dataset_validation.json`, `data/classification/dataset_metadata.json`

### Classification (`data/classification/dataset.csv`)

| Check | Result |
|-------|--------|
| Total samples | **91** |
| Unique source templates | **21** |
| Labels | All five required labels; **0** invalid |
| Duplicate sample IDs | **0** |
| Duplicate code bodies | **0** |
| AST syntax failures | **0** |
| Behavioral re-check (template tests) | **0** mismatches |
| Template split leakage | **Checked — none** (grouped by `source_template_id`, seed **42**) |
| SHA256 | `c7ae7a8d836b64c9aa139e9c1fe77724329840a5629f87a08b3a2d1e993aff37` |

**Class distribution:** correct 42, wrong_operator 16, wrong_initialization 12, wrong_condition 12, off_by_one_loop 9.

**Splits:** train 61, val 14, test 16 (class counts per split in validation JSON).

**Limitation:** Splits are **template-grouped**, not class-stratified. With 91 samples, meaningful grouped+stratified splits are not feasible; this is documented in validation output.

### Similarity (`data/similarity/pairs.csv`)

| Check | Result |
|-------|--------|
| Total pairs | **193** (61 similar / 132 dissimilar) |
| Duplicate pairs | **0** |
| Endpoint / tag consistency | **Repaired 2026-10-19** (see below) |
| SHA256 (after repair) | `920f5abb101cf241396c1bfabd56960418c3adec8b6c8911cfc0eb839acc6767` |

**Repair:** Prior `pairs.csv` tagged **91** pairs with a single endpoint’s split (leakage). Regenerated via `py -3.13 -m training.regenerate_similarity_pairs` (backup: `backups/pairs_20261019_110004.csv`).

| Split tag | Count |
|-----------|------:|
| train | 91 |
| cross_split (excluded from eval) | 91 |
| val | 4 |
| test | 7 |

**Evaluable pairs (same-split endpoints only):** **102**. **Test pairs for benchmarking:** **7** — too small for stable conclusions; retained as **smoke-test** corpus only.

---

## 2. Model evaluation (2026-10-19, post-repair)

### Classification baseline (TF-IDF fit on **train only**, test **N=16**)

| Metric | Value |
|--------|------:|
| Accuracy | 43.75% |
| Macro F1 | **27.17%** |

Artifacts: `baseline_classification_metrics.json`, `baseline_classification_report.json`, `baseline_confusion_matrix.json`, `baseline_eval_config.json`, figure `artifacts/figures/baseline_classification_confusion.png`.

### CodeBERT classifier (prior 5-epoch CPU run, test **N=16**)

| Metric | Value |
|--------|------:|
| Accuracy | 50.00% |
| Macro F1 | 13.33% |

Baseline **macro F1 remains higher** on this test split.

### Similarity (val threshold → test metrics, **test N=7**)

Method: grid 0.1–0.9 step 0.05, maximize F1 on **val** (4 pairs); scores are cosine similarity, **not** probabilities. ROC-AUC computed on test where both classes present.

| Model | Val threshold | Test F1 | Test ROC-AUC |
|-------|--------------|--------:|-------------:|
| TF-IDF | 0.55 | **80.00%** | 1.0 |
| CodeBERT embeddings | 0.10 | 72.73% | 0.917 |

Artifacts: `similarity_metrics.json`, `similarity_threshold_config.json`, `tfidf_similarity_test_report.json`, `codebert_similarity_test_report.json`, binary confusion JSON files.

**Do not** compare these to pre-repair metrics that used **38** mis-tagged test pairs.

---

## 2b. benchmark_v2 (Phase 2 dataset, 2026-10-19)

**Generate:** `py -3.13 -m training.dataset_pipeline.generate_benchmark`  
**Docs:** `docs/BENCHMARK_V2.md`  
**Checksum:** `6d2d6ab93918ef86ebbb36d46b0ced5f109dbc0983f2229a067fe72ef194ba55` (seed **42**)

| Item | Value |
|------|------:|
| Classification samples | **121** (accepted; 4 rejected; 10 in review queue) |
| Independent program **families** | **20** (train 12 / val 3 / test 5 families) |
| Similarity pairs (total) | **659** |
| Evaluable pairs (non–cross_split) | **338** (test **58**, val **20**) |
| human_reviewed | **false** (automated labels + finite behavioral tests) |

**Baseline on v2 test (N=31, TF-IDF fit train-only):** accuracy 38.71%, macro F1 **33.99%** (`artifacts/metrics/benchmark_v2/20261019_062841/`).  
**Similarity TF-IDF test (N=58):** F1 **68.09%**, ROC-AUC **0.764** (val-tuned threshold).  
**Hybrid TF-IDF+static:** macro F1 **18.41%** on same test — reported separately; heuristics are not ground truth.

Legacy v1 paths and production checkpoint **unchanged**. CodeBERT on v2: `py -3.13 -m training.train_codebert_benchmark_v2` (writes under `models/benchmark_v2/` only).

---

## 3. Software tests

```
py -3.13 -m pytest backend/tests -q
# 56 passed (2026-10-19)
```

Includes `backend/tests/test_dataset_validation.py` (labels, leakage, pairs, threshold methodology, missing files).

---

## 4. Application stack

| Component | Status |
|-----------|--------|
| FastAPI backend | Working |
| React + Vite frontend | Build OK |
| Baseline joblibs | Retrained on validated data |
| CodeBERT classifier weights | Local `model.safetensors` (gitignored) |

---

## 5. Commands

```powershell
py -3.13 -m training.validate_dataset
py -3.13 -m training.regenerate_similarity_pairs   # if pairs stale
py -3.13 -m training.train_baselines
py -3.13 -m training.evaluate_similarity
py -3.13 -m pytest backend/tests -q
```

---

## 6. CodeBERT classifier experiment (2026-10-19, controlled)

**Production checkpoint preserved:** `models/codebert_classifier/` (not overwritten).  
**Backup:** `backups/production_codebert_20261019_110811/`  
**Experiment:** `models/experiments/codebert_classifier_20261019_4ep/`  
**Metrics:** `artifacts/metrics/experiments/codebert_classifier_20261019_4ep/`

| Setting | Value |
|---------|--------|
| Hardware | **CPU** (CUDA unavailable) |
| Epochs | **4** |
| LR | 2e-5 |
| Batch size | 8 |
| Max length | 128 |
| Seed | 42 |
| Checkpoint rule | Best **validation macro F1** |
| Best epoch | **4** |
| Best val macro F1 | **0.2705** |

**Test set (N=16, single final eval):**

| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| Experiment CodeBERT | 43.75% | **28.00%** |
| TF-IDF + LogReg baseline | 43.75% | 27.17% |

CodeBERT macro F1 is **marginally higher** (+0.83 pp) on this tiny test split; difference is **not statistically meaningful**. Production checkpoint **unchanged**; API still serves `models/codebert_classifier/`.

Pre-flight: `py -3.13 -m training.verify_classifier_training`  
Colab GPU path: `training/COLAB_CODEBERT_CLASSIFIER.md`

---

## 7. Release readiness (2026-10-19 final QA)

### API model loading (verified)

| Endpoint | Implementation | Weights / model |
|----------|----------------|-----------------|
| `POST /api/classification/analyze` | `classification_service.analyze_code` | **`models/codebert_classifier/`** if load succeeds; else `models/baselines/` |
| `POST /api/similarity/analyze` | `similarity_service.analyze_similarity` | **`microsoft/codebert-base`** encoder + `similarity_tfidf_vectorizer.joblib` |

Checkpoint `label_mapping.json` **matches** `backend/app/config.py` `LABEL_TO_ID`.

### Commands run (this gate)

| Command | Result |
|---------|--------|
| `py -3.13 -m pytest backend/tests -q` | **57 passed** |
| `py -3.13 -m training.validate_dataset` | **Valid** |
| `py -3.13 -m training.verify_classifier_training` | **OK** |
| `py -3.13 -m training.train_baselines` | **OK** |
| `py -3.13 -m training.evaluate_similarity` | **OK** |
| `cd frontend; npm run build` | **OK** |

### UI / metrics caveat

Evaluation tab JSON (`artifacts/metrics/codebert_classification_metrics.json`) reflects **production** training (macro F1 **13.33%**, N=16). Experiment macro F1 **28.00%** is under `artifacts/metrics/experiments/codebert_classifier_20261019_4ep/`.

Similarity test **N=7** — preliminary only.

Demo: `scripts/DEMO.ps1`

---

## 8. Phase 5 — benchmark_v3 (2026-10-19)

**Audit:** `py -3.13 -m training.phase5_audit` → `artifacts/metrics/phase5/audit_report.json`  
**Generate v3:** `py -3.13 -m training.dataset_pipeline.generate_benchmark_v3`  
**Review CLI:** `py -3.13 -m training.label_review_cli list` → `data/benchmark_v2/review_decisions.json` (all **pending**)

| Item | benchmark_v3 (generated) |
|------|--------------------------|
| Samples | **190** |
| Families | **53** (train 31 / val 7 / test 15) |
| Checksum | `9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383` |
| Val class coverage | all 5 labels present (incl. **off_by_one_loop**: 2) |
| TF-IDF val macro F1 (runner) | **26.89%** (N=30) |
| Synthetic / human reviewed | unreviewed synthetic; `human_reviewed: false` |

benchmark_v2 checksum **unchanged**. Production checkpoint **unchanged**. GPU matrix in `artifacts/metrics/benchmark_v3/gpu_readiness.json` — not auto-run.

---

## 9. Phase 6 — benchmark_v3 CodeBERT matrix (2026-10-19)

**Runner:** `py -3.13 -m training.run_v3_codebert_experiments` (CPU run with `--allow-cpu`; no GPU on dev machine)  
**Selected (validation only):** `gpu_3_balanced` — val macro F1 **34.59%** vs TF-IDF val **26.89%**  
**Final test (single post-selection):** macro F1 **27.13%** — synthetic labels, not retuned in Phase 7  
**Artifacts:** `artifacts/metrics/benchmark_v3/codebert_experiments/`  
**Production / v2:** unchanged

---

## 10. Phase 7 — error analysis & review (2026-10-19)

**Analysis:** `py -3.13 -m training.phase7_error_analysis`  
**Report:** `artifacts/metrics/phase7/error_analysis_report.json`, `docs/PHASE7_ERROR_ANALYSIS.md`  
**Review queue:** `data/benchmark_v3/review_decisions.json` — **30 items, all pending** (no human decisions recorded; priority batch covers 4 samples)  
**Reviewed eval slice:** **blocked** (`data/benchmark_reviewed/manifest.json`)  
**API:** `/evaluation/phase7-error-analysis`, `/evaluation/error-explorer`, `/evaluation/review-queue`  
**Classifier:** optional `problem_statement`; diagnostics distinguish model vs static heuristics; tests not run by default  
**Tests:** 91 passed; frontend build OK

---

## 11. Phase 8 — human adjudication workflow (2026-10-19)

**Packets:** `data/benchmark_v3/review_packets/priority_batch.json` (S0027, S0028, S0003, S0009)  
**Fixtures:** `data/benchmark_v3/review_fixtures/`  
**CLI:** `packet`, `verify`, `decide` on `label_review_cli --dataset benchmark_v3`  
**Behavioral tests:** `py -3.13 -m training.review_test_runner --all-priority` (allowlist only)  
**Decisions recorded:** **0** — all 40 items still `pending`  
**Reviewed slice:** **blocked** (`build_reviewed_eval_slice` exit 0, count 0)  
**Docs:** `docs/PHASE8_HUMAN_ADJUDICATION.md`

---

## 12. Remaining work (prioritized)

1. **Human** adjudication of priority batch V3RQ001–004; then `build_reviewed_eval_slice`.
2. Fix generator/template cases where both models fail (e.g. S0003, S0027, S0028).
3. Expand similarity test split beyond 7 pairs.
4. GPU replay of Phase 6 after label quality improves (optional).
5. FastAPI lifespan migration.
