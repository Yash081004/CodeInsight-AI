# Experiment matrix, cost estimates, acceptance criteria

**No experiments were executed in Phase 9.** This is a proposal only. All runs must write to **non-production** paths (e.g. `models/experiments/phase9_*`, `artifacts/metrics/phase9/experiments/`).

## Acceptance criteria (production promotion gate)

A candidate **must not** replace `models/codebert_classifier/` unless **all** of the following hold on **benchmark_v3 val** (selection) with final numbers reported once on **benchmark_v3 test** (single primary holdout, not used during selection):

| Criterion | Minimum | Rationale |
|-----------|---------|-----------|
| Macro F1 | ≥ **0.35** | Must beat TF-IDF (~0.27 on v1; ~0.34 on v2 test) with margin |
| Per-class fault recall | each fault class ≥ **0.25** on val | No collapse; at least one correct detection per class |
| `correct` recall | ≥ **0.50** | Avoid predicting fault everywhere |
| Max single-class prediction share on val | ≤ **70%** | Anti-collapse guard |
| Label–test audit | 0 unresolved **verified** queue items in training set | Or exclude flagged sample IDs from train |

**Disqualifiers:** tuning hyperparameters on v3 test; any change to historical benchmark labels; val F1 improved only by spec rules (report neural-only vs hybrid separately).

Report **neural-only** metrics separately from spec-assisted API routing.

## Ranked root causes (summary)

See [MODEL_FAILURE_ANALYSIS.md](./MODEL_FAILURE_ANALYSIS.md). Top drivers: **(1)** class imbalance / collapse, **(2)** tiny n vs CodeBERT, **(3)** code-only inputs, **(5)** synthetic label noise.

## Proposed experiment matrix (CPU-realistic)

Observed reference times (repo logs / prior runs): CodeBERT ~1–4 min/epoch on CPU for n≈100–190, batch 4; TF-IDF baseline **seconds**.

| ID | Data | Input | Training change | Output dir | Est. CPU time |
|----|------|-------|-----------------|------------|---------------|
| **E0** | benchmark_v3 | code only | TF-IDF + LogReg (reproduce) | `artifacts/metrics/phase9/experiments/E0_tfidf_v3/` | **< 1 min** |
| **E1** | benchmark_v3 minus review_queue templates | code only | CodeBERT 3 ep, weighted CE, max_len 256, `--output_dir models/experiments/phase9_e1_codebert_v3` | same | **~15–45 min** |
| **E2** | E1 data | code only | + `--balanced_sampler` (keep class weights off) | `phase9_e2` | ~same as E1 |
| **E3** | E1 data | `[SPEC] {expected_behavior}\n[CODE] {code}` truncated | CodeBERT 3 ep, weighted CE | `phase9_e3` | ~same as E1 |
| **E4** | E1 data | code only | Linear probe: freeze CodeBERT 2 ep then head-only 2 ep | `phase9_e4` | ~20–50 min |

**Not in initial matrix (defer):** full v3 re-generation, human relabeling sprint, GPU multi-seed sweeps, similarity encoder fine-tuning.

### E0 rationale

Establish **strong baseline on v3 val** before any CodeBERT run. If E0 macro F1 ≥ 0.35 on val, neural runs must beat it clearly.

### E1 rationale

Uses **largest clean-ish family split** (53 families) while **excluding or relabeling** the 10 review-queue templates from **train only** (IDs from queue file — no CSV label edits; drop rows matching template_id).

### E3 rationale

Tests whether **expected_behavior** (already in CSV) helps without API spec rules — still synthetic text, but closer to user-facing spec.

## Data quality work (no training)

| Task | Effort | Value |
|------|--------|-------|
| Human review of 10 queue items + 5 fixtures | hours | Fixes verified label noise |
| Expand review_queue processing in `label_review_cli` | low | Document accept/reject |
| Add `phase9_dataset_audit` to CI | low | Regression on checksums |

## Single highest-value next experiment

**Run E0 + E1 together:**

1. **E0** — TF-IDF on benchmark_v3 (val selection metric) — cheap, confirms baseline on the same split policy as future CodeBERT.  
2. **E1** — CodeBERT on v3 **with review-queue templates excluded from train/val** (filter in loader, not label edits), 3 epochs, `max_length=256`, output under `models/experiments/phase9_e1_*`.

If E1 still shows >70% `correct` predictions on val, **stop** before E2–E4; invest in **label review (queue)** and **problem-spec inputs (E3)** instead of more epochs.

## Commands (when approved — not run in Phase 9)

```powershell
# E0 — extend train_baselines.py with --data-path data/benchmark_v3/classification/dataset.csv
# E1 — example (custom output dir required):
py -3.13 -m training.train_codebert_classifier ^
  --data-path data/benchmark_v3/classification/dataset.csv ^
  --output_dir models/experiments/phase9_e1_codebert_v3 ^
  --epochs 3 --max_length 256 --batch_size 4
```

Add `--exclude-templates-from data/benchmark_v3/classification/review_queue.json` in a small follow-up PR before executing E1 (filter only; no label changes).
