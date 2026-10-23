# CodeInsight AI

Deep learning–based **semantic code similarity** (CodeBERT + TF-IDF) and **five-class logical error classification** for beginner Python programs.

**Status:** See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for verified metrics and limitations.  
**Audit history:** [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).

## Requirements

- **Python 3.13** (`py -3.13`) — backend, training, tests  
- **Node.js 18+** — frontend  
- Fine-tuned classifier weights: `models/codebert_classifier/model.safetensors` (gitignored; train locally or restore from backup)

## Quick start

From the repository root:

```powershell
# Backend (terminal 1)
py -3.13 -m pip install -r backend/requirements.txt
py -3.13 -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Frontend (terminal 2)
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173 (Vite proxies `/api` to the backend when configured in `vite.config.js`).

Production UI build:

```powershell
cd frontend
npm run build
npm run preview
```

## Models

| Role | Path |
|------|------|
| **Production classifier (API)** | `models/codebert_classifier/` via `Settings.classifier_model_path` |
| **Similarity encoder** | Hugging Face `microsoft/codebert-base` (cached after first use) |
| **Baselines** | `models/baselines/*.joblib` |
| **Controlled experiment (not served by default)** | `models/experiments/codebert_classifier_20261019_4ep/` |

Classification API tries CodeBERT first, then TF-IDF + Logistic Regression.

## benchmark_v2 (Phase 2)

Stronger, versioned benchmark with **family-isolated splits** and expanded similarity pairs. Legacy v1 data under `data/classification/` is unchanged.

```powershell
py -3.13 -m training.dataset_pipeline.generate_benchmark
py -3.13 -m training.train_eval_benchmark_v2
py -3.13 -m training.train_codebert_benchmark_v2 --epochs 4 --batch_size 4
```

Details: [`docs/BENCHMARK_V2.md`](docs/BENCHMARK_V2.md).

## Dataset (validated, legacy v1)

- **Classification:** 91 samples, 21 templates, checksum `c7ae7a8d…993aff37`, split seed **42** (grouped by template)  
- **Similarity:** 193 pairs; **91** `cross_split` (excluded); **7** test pairs for benchmarking — **preliminary only**

## Reproduction commands

```powershell
py -3.13 -m training.validate_dataset
py -3.13 -m training.verify_classifier_training
py -3.13 -m training.train_baselines
py -3.13 -m training.evaluate_similarity
py -3.13 -m pytest backend/tests -q
```

### Automated QA (full stack)

With **one** backend on `127.0.0.1:8000` and Vite on `5173` (or pass `--start-frontend`):

```powershell
py -3.13 scripts/run_qa.py --start-frontend
# or
.\scripts\run_qa.ps1 --start-frontend
```

Reports land in `test-results/qa_<timestamp>/` (`qa_report.json` + `qa_summary.md`). Model-quality warnings are separated from software pass/fail.

**Phase 9 (dataset / model audit):** [`docs/phase9/README.md`](docs/phase9/README.md) — `py -3.13 -m training.phase9_dataset_audit`

**Phase 10 (E0/E1 v3 experiments):** [`docs/phase10/README.md`](docs/phase10/README.md) — `py -3.13 -m training.phase10_experiments`

**Phase 11 (E3 spec+code comparison):** [`docs/phase11/README.md`](docs/phase11/README.md) — `py -3.13 -m training.phase11_experiments`

**Phase 12 (dataset readiness, no training):** [`docs/phase12/README.md`](docs/phase12/README.md) — `py -3.13 -m training.phase12_run`

Do **not** run `train_codebert_classifier` without `--output_dir` if you intend to preserve the production checkpoint.

## Docker (local)

Requires Docker Desktop. Mounts `data/`, `models/`, `artifacts/` read-only into the backend.

```powershell
docker compose build
docker compose up -d
# UI: http://localhost:8080  API: http://localhost:8000/api/health
curl http://127.0.0.1:8000/api/ready
```

Ensure `models/codebert_classifier/model.safetensors` exists locally before building (not in git).

## Demonstration

See [`scripts/DEMO.ps1`](scripts/DEMO.ps1) for a scripted walkthrough using live API responses.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
- [`docs/ACADEMIC_REPORT_OUTLINE.md`](docs/ACADEMIC_REPORT_OUTLINE.md)

## Classification comparison (test N=16)

| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| TF-IDF + Logistic Regression | 43.75% | **27.17%** |
| Experimental CodeBERT (4 epochs, val F1 selection) | 43.75% | **28.00%** |

The **0.83** percentage-point macro F1 difference is **inconclusive** on this tiny test set. **Production checkpoint unchanged.**

Experimental artifacts: `artifacts/metrics/experiments/codebert_classifier_20261019_4ep/`

Evaluation tab JSON under `artifacts/metrics/codebert_classification_metrics.json` reflects the **production** training run unless you update that file manually.
