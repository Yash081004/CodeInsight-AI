# CodeBERT Classifier — Google Colab (GPU) Workflow

Use when local CPU training is too slow. Reuses the same script and validated dataset.

## 1. Upload or clone project

```python
!git clone <your-repo-url> CodeInsight-AI
%cd CodeInsight-AI
```

Or upload `data/classification/dataset.csv` and copy training modules.

## 2. Install dependencies

```python
!pip install -q torch transformers pandas scikit-learn
```

## 3. Verify dataset (checksum must match validated corpus)

```python
!python -m training.verify_classifier_training
```

Expected checksum prefix: `c7ae7a8d` … suffix `aff37`.

## 4. Train experiment (does not overwrite production path)

```python
!python -m training.train_codebert_classifier \
  --epochs 4 \
  --batch_size 8 \
  --lr 2e-5 \
  --max_length 128 \
  --seed 42 \
  --output_dir models/experiments/codebert_classifier_colab \
  --metrics_dir artifacts/metrics/experiments/codebert_classifier_colab
```

## 5. Download artifacts

- `models/experiments/codebert_classifier_colab/` (checkpoint)
- `artifacts/metrics/experiments/codebert_classifier_colab/` (metrics, confusion matrix)

Do **not** replace `models/codebert_classifier/` unless `classification_model_comparison.json` shows a justified improvement over baseline macro F1 on test **N=16**.
