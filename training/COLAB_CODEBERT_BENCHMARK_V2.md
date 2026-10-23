# CodeBERT benchmark_v2 (Colab / GPU)

Production checkpoint is **not** modified. Outputs go to `models/benchmark_v2/codebert_<run_id>/`.

```python
!git clone <your-repo> CodeInsight-AI
%cd CodeInsight-AI
!pip install -r backend/requirements.txt
```

```bash
python -m training.train_codebert_benchmark_v2 \
  --epochs 4 \
  --batch_size 8 \
  --lr 2e-5 \
  --eval_batch_size 16
```

After training, gallery artifacts are generated automatically under
`artifacts/metrics/benchmark_v2/codebert_<run_id>/`.

Optional improvement experiment (validation only for selection):

```bash
python -m training.train_codebert_benchmark_v2 \
  --epochs 3 --lr 3e-5 --batch_size 8
```

Compare **validation macro F1** in each run's `training_history.json` before evaluating test once.
