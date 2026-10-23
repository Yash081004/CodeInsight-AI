# CodeInsight AI — Implementation Plan (Ultimate Platform)

## Stage 1 — Audit & architecture ✅ (baseline)

| Acceptance | Status |
|------------|--------|
| Documented frontend→API→model flows | `docs/ARCHITECTURE.md` |
| Production vs experiment checkpoints preserved | Verified |
| No automatic experiment promotion | Policy enforced |

## Stage 2 — Dataset & model quality 🔄

| Acceptance | Status |
|------------|--------|
| Validated 91-sample corpus + checksum | ✅ `training.validate_dataset` |
| Leakage-safe similarity splits | ✅ 91 cross_split excluded |
| Expanded diverse templates (+holdout v2) | ⏳ Planned — requires regeneration + retrain |
| New isolated final test set | ⏳ Not created (avoid silent relabel) |
| Measured accuracy claims only | ✅ |

**Next:** Add 10+ new program families in `generate_dataset.py`, regenerate, re-baseline, optional GPU CodeBERT retrain.

## Stage 3 — Premium UI/UX 🔄

| Acceptance | Status |
|------------|--------|
| Dark + light theme | ✅ Header toggle |
| Evaluation run selector (no mixed metrics) | ✅ Evaluation tab |
| Classifier shows API probabilities + diagnostics | ✅ Fixed schema binding |
| Side-by-side editors / charts polish | ⏳ Incremental |

## Stage 4 — Explainable AI ✅ (v1)

| Acceptance | Status |
|------------|--------|
| Prediction vs confidence vs static findings | ✅ `diagnostics` on classification API |
| No attention-based causal claims | ✅ |
| Advisory suggestions only | ✅ |

## Stage 5 — Evaluation lab ✅ (v1)

| Acceptance | Status |
|------------|--------|
| `/api/evaluation/experiments` | ✅ |
| Per-run summary + confusion matrix query params | ✅ |
| Similarity marked preliminary (N=7) | ✅ UI + API |

## Stage 6 — Backend hardening 🔄

| Acceptance | Status |
|------------|--------|
| `/api/ready` | ✅ |
| Input validation (length, blank) | ✅ existing |
| Structured logging without raw code | ✅ (history uses hash) |
| Request body size limit | ⏳ nginx/docker 512k; add Starlette limit optional |

## Stage 7 — Deployment 🔄

| Acceptance | Status |
|------------|--------|
| Dockerfile.backend + Dockerfile.frontend | ✅ |
| docker-compose.yml + healthcheck | ✅ |
| Public cloud deploy | ❌ No credentials — local container verify only |

## Stage 8 — QA 🔄

Run before release tag:

```powershell
py -3.13 -m pytest backend/tests -q
cd frontend && npm run build
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/api/ready
```

## Stage 9 — Deliverables 🔄

| Item | Path |
|------|------|
| Technical README | `README.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Academic outline | `docs/ACADEMIC_REPORT_OUTLINE.md` |
| Demo script | `scripts/DEMO.ps1` |
