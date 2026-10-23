# Phase 10 — E0 + E1 controlled experiments (benchmark v3)

Executed **2026-10-20**. Production checkpoint **unchanged** (`models/codebert_classifier/`).

## Artifacts

| Experiment | Metrics | Models |
|------------|---------|--------|
| **E0** TF-IDF | [`artifacts/metrics/phase10/E0_tfidf_v3/`](../../artifacts/metrics/phase10/E0_tfidf_v3/) | [`models/experiments/phase10_e0_tfidf_v3/`](../../models/experiments/phase10_e0_tfidf_v3/) |
| **E1** CodeBERT | [`artifacts/metrics/phase10/E1_codebert_v3/`](../../artifacts/metrics/phase10/E1_codebert_v3/) | [`models/experiments/phase10_e1_codebert_v3/`](../../models/experiments/phase10_e1_codebert_v3/) |

- Acceptance: [`ACCEPTANCE_REPORT.md`](./ACCEPTANCE_REPORT.md), [`artifacts/metrics/phase10/acceptance_report.json`](../../artifacts/metrics/phase10/acceptance_report.json)
- E1 exclusions: [`E1_codebert_v3/exclusion_manifest.json`](../../artifacts/metrics/phase10/E1_codebert_v3/exclusion_manifest.json)

## Reproduce

```powershell
py -3.13 -m training.phase10_experiments --e0
py -3.13 -m training.phase10_experiments --e1 --acceptance-report
```

## Results summary

| | E0 val macro F1 | E0 test macro F1 | E1 val macro F1 | E1 test macro F1 |
|--|-----------------|------------------|-----------------|------------------|
| **Score** | 0.327 | 0.302 | 0.211 | 0.151 |
| **Gate** | **FAIL** | (report only) | **FAIL** | (report only) |

Neither experiment meets the Phase 9 promotion gate on validation.

## Recommendation

**Next: E3 (spec + code input) or human review of the v3 review queue** — E1 still collapsed to 100% `correct` on validation after queue exclusion. Label-quality fixes alone are unlikely sufficient without giving the model `expected_behavior` (or problem statements) at training time. Prefer **E3** on the same filtered view before more epochs or larger CodeBERT runs on CPU.
