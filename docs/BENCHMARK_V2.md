# benchmark_v2 — Dataset and evaluation

Versioned benchmark under `data/benchmark_v2/` (does **not** replace legacy `data/classification/`).

## Generate

```powershell
py -3.13 -m training.dataset_pipeline.generate_benchmark
py -3.13 -m training.verify_benchmark_dataset
```

Outputs:

| Path | Purpose |
|------|---------|
| `classification/dataset.csv` | 5-class samples with `family_id`, mutation provenance |
| `similarity/pairs.csv` | Typed pairs (`pair_type`, `family_a`, `family_b`) |
| `manifest.json` | Seed, checksum, split manifest |
| `quality_report.json` | Reproducible quality summary |
| `review_queue.json` | Ambiguous / failed behavioral checks (not human-reviewed by default) |

## Splits

Families are assigned wholly to **train / val / test** (seed 42). Similarity pairs with endpoints in different splits are tagged `cross_split` and excluded from threshold tuning and test reporting.

**Limitation:** Independent program families are finite (~20 in current generator). Metrics on test families are indicative, not a large-scale holdout.

## Train / evaluate (no production checkpoint)

```powershell
# TF-IDF + LogReg + similarity TF-IDF + hybrid static system
py -3.13 -m training.train_eval_benchmark_v2

# CodeBERT (writes under models/benchmark_v2/codebert_<run_id>/)
py -3.13 -m training.train_codebert_benchmark_v2 --epochs 4 --batch_size 4
```

Artifacts: `artifacts/metrics/benchmark_v2/<run_id>/`

Evaluation UI/API: select run id `benchmark_v2/<run_id>`.

## Phase 3 — CodeBERT v2 run (2026-10-19, CPU 2 epochs)

| Model | Test N | Families | Macro F1 | Accuracy |
|-------|-------:|---------:|---------:|---------:|
| TF-IDF + LogReg (`20261019_062841`) | 31 | 5 | **33.99%** | 38.71% |
| CodeBERT (`codebert_20261019_082413`, best epoch **1**, val macro F1 **19.23%**) | 31 | 5 | **14.69%** | 58.06% |

CodeBERT collapsed toward predicting `correct` (18/18 correct-class recall in report); macro F1 is **below** the TF-IDF baseline on the same test IDs. Gallery: `artifacts/metrics/benchmark_v2/codebert_20261019_082413/classification_error_gallery.json`.

Suggested next experiment (val-only selection): `--lr 3e-5 --epochs 4` on GPU (see `training/COLAB_CODEBERT_BENCHMARK_V2.md`).

## Phase 4 — collapse audit and ablations

```powershell
py -3.13 -m training.benchmark_v2_collapse_audit
py -3.13 -m training.run_benchmark_v2_ablations --cpu_matrix
py -3.13 -m training.finalize_ablation_test_eval
```

Audit: `artifacts/metrics/benchmark_v2/phase4/collapse_audit.json`  
Ablations: `artifacts/metrics/benchmark_v2/ablations/`  
Evaluation Lab: **Phase 4 ablation comparison** table (`GET /api/evaluation/ablations`).

**Findings (evidence-based):** Phase-3 checkpoint predicts `correct` for 100% of val and test predictions; softmax entropy ≈1.6 (near uniform). TF-IDF val macro F1 **22.22%** vs CodeBERT ablations **19.23%** (all tied). Unweighted CE and LR 3e-5 did not improve val F1 on 2 CPU epochs. Test set is a **previously inspected** benchmark — not used for ablation selection.

## Phase 5 pointer

benchmark_v2 is **frozen** (checksum unchanged). New work uses **benchmark_v3** — see [`BENCHMARK_V3.md`](BENCHMARK_V3.md) and `py -3.13 -m training.phase5_audit`.

## Concepts

- **Similarity label 1** = same-family correct pair or documented relation — not proven semantic equivalence.
- **Behavioral tests** = finite template tests; passing does not prove universal correctness.
- **human_reviewed: false** in quality report unless items are explicitly cleared from `review_queue.json`.
