# Model failure analysis

Evidence from checkpoints, training code, and frozen metrics artifacts (no retraining in this phase).

## Reproduced metrics (same splits as artifacts)

### Legacy v1 — test (n=16), checksum `c7ae7a8d…`

| Model | Macro F1 | Macro recall | Notes |
|-------|----------|--------------|--------|
| **Production CodeBERT** | **0.1333** | 0.20 | `artifacts/metrics/codebert_classification_metrics.json` |
| TF-IDF + LogReg | **0.2717** | 0.325 | `artifacts/metrics/baseline_classification_metrics.json` |

**CodeBERT confusion matrix** (`artifacts/metrics/confusion_matrix.json`): **all 8 fault test samples predicted as `correct`** (100% test recall for correct, **0% recall for every fault class**). This is full **class collapse** on held-out v1 test.

**TF-IDF:** non-zero recall on wrong_operator (0.5) and wrong_condition (0.5); still 0 on wrong_init and off_by_one.

### benchmark_v2 — test (n=31)

| Model | Macro F1 |
|-------|----------|
| CodeBERT (v2 run) | 0.1469 |
| TF-IDF | 0.3399 |

CodeBERT again **100% recall on `correct`**, **0% on all fault classes** in per-class metrics artifact (`benchmark_v2/codebert_20261019_082413/codebert_classification_metrics.json`).

### Similarity (legacy test)

- **7** held-out pairs; hybrid F1 **0.727** with **3 false positives** (`similarity_hybrid_policy.json`).
- CodeBERT-only cosine **~0.99** on sort-vs-search diagnostic; hybrid policy rejects (product fix, not encoder fix).

## Production checkpoint configuration

`models/codebert_classifier/training_config.json`:

- `max_length`: **128**
- `best_epoch`: **1**
- `best_val_f1`: **0.12** (legacy v1 val)
- Trained with default **weighted cross-entropy** (`train_codebert_classifier.py`)

Training uses **code string only** — no `problem_statement`, no `expected_behavior` (`CodeDataset` tokenizes `code` column only). The model **cannot** learn spec-conditioned faults from current training.

## Why collapse toward `correct` is likely (ranked mechanisms)

### 1. Severe class imbalance + majority-class optimum (high confidence)

Train split ~46% correct; fault classes split five ways (~6–11 each). With small data, predicting **correct** maximizes accuracy and can yield **lower loss** than rare classes unless weighting/sampling fully compensates.  
**Evidence:** collapse on both v1 and v2 test; weighted CE still yields 0 fault recall in production eval.

### 2. Dataset scale vs model capacity (high confidence)

~61 train samples for CodeBERT (~125M params) is far below typical fine-tuning needs.  
**Evidence:** val macro F1 **0.12** at best epoch 1; no sustained val improvement in config snapshot.

### 3. Code-only input — under-specified task (high confidence)

Many faults are **not identifiable from syntax alone** (e.g. wrong_operator vs correct without I/O semantics). API spec-assist succeeds on diagnostic cases where **problem text** is present; CodeBERT never sees that text in training.  
**Evidence:** `error_classifier.predict(code)` only; diagnostic API uses spec rules when collapsed.

### 4. Short `max_length=128` (medium confidence)

Beginner programs usually fit; truncation is unlikely to be the primary collapse driver but limits future multi-file/spec concatenation experiments.

### 5. Label noise / weak supervision (medium confidence)

benchmark_v3 **review_queue** proves some labels are **inconsistent with generator tests**. Noisy labels push decision boundaries toward “always correct” under CE.  
**Evidence:** 10× `mutation_still_passes_tests` entries.

### 6. Template memorization failure (medium confidence)

Grouped splits withhold whole templates; test templates are unseen. Model must generalize to new template surface forms — **not** just memorize. TF-IDF outperforms CodeBERT, suggesting **simple n-grams carry more signal** than learned embeddings at this n.

### 7. Checkpoint selection on weak val (medium confidence)

`best_epoch=1` with val F1 0.12 saved as production — selection signal is noisy; still, later epochs did not show strong fault recall in v2 ablations (~0.19 val macro F1 ceiling).

## Training pipeline facts (code references)

| Component | Behavior |
|-----------|----------|
| Loss | `CrossEntropyLoss(weight=inverse_freq)` optional; default on |
| Sampler | Optional `WeightedRandomSampler` (`--balanced_sampler`); **not** production default |
| Optimizer | AdamW lr 2e-5, grad clip 1.0 |
| Early stopping | Optional patience; production config shows epoch 1 best |
| Eval metric | Macro F1 on val for checkpoint |
| Baseline | TF-IDF 5k features, bigrams, LogReg `class_weight=balanced` |

## Prediction distribution

Legacy v1 test CodeBERT: **predicted class = correct for 16/16** (confusion matrix). This is not random — it is **systematic collapse**.

## Similarity (separate from classifier)

Encoder similarity is **not** semantic equivalence for algorithmically different code; hybrid policy is a **decision layer**, not retrained encoder. Do not use 7-pair test F1 to justify encoder quality.
