# CodeBERT benchmark_v3 GPU training (Colab)

Local preflight reported **no CUDA** on Windows; use this notebook flow for the Phase 6 matrix on GPU.

## Dataset

- Path: `data/benchmark_v3/classification/dataset.csv`
- Checksum: `9c3220f9a16575c5cfef17d5622d94676a31759c10cef555b9934ee85c90d383`
- Do **not** overwrite `models/codebert_classifier/`.

## Commands (GPU runtime)

```bash
pip install -r backend/requirements.txt
python -m training.run_v3_codebert_experiments --epochs 4 --patience 1 --seed 42
python -m training.run_v3_codebert_experiments --skip-train --final-test
```

Copy artifacts back:

- `artifacts/metrics/benchmark_v3/codebert_experiments/`
- `models/benchmark_v3/codebert_experiments/`

Without `--allow-cpu`, the runner exits if CUDA is unavailable.
