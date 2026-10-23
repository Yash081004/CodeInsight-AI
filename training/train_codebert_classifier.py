"""
CodeInsight AI - CodeBERT Fine-Tuning Script

Fine-tunes microsoft/codebert-base for 5-class logic error classification.
Designed to run on CPU (slow) or GPU (recommended).
Can be run locally or in Google Colab.

Usage:
    python -m training.train_codebert_classifier
    python -m training.train_codebert_classifier --epochs 3 --batch_size 8 --lr 2e-5
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL


class CodeDataset(Dataset):
    """PyTorch dataset for tokenized text samples (code-only or spec+code)."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(args):
    """Main training function."""
    print("=" * 60)
    print("CodeInsight AI - CodeBERT Fine-Tuning")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cpu":
        print("\n[WARNING] Training on CPU will be slow. Consider using Google Colab with GPU.")
        print("   For a quick test, use --epochs 1 --batch_size 4\n")
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run: python -m training.generate_dataset")
        return
    
    from training.dataset_validation import sha256_file
    from training.verify_classifier_training import verify as verify_training_data

    if args.skip_preflight:
        print("[WARN] Skipping dataset pre-flight checks.")
    else:
        ok, preflight_errors = verify_training_data(data_path)
        if not ok:
            from training.verify_benchmark_dataset import verify as verify_benchmark

            manifest_path = data_path.parent.parent / "manifest.json"
            ok_b, bench_errors = verify_benchmark(
                data_path,
                manifest_path if manifest_path.exists() else None,
            )
            if not ok_b:
                print("ERROR: Training pre-flight checks failed (legacy v1 and benchmark_v2):")
                for err in preflight_errors + bench_errors:
                    print(f"  - {err}")
                return
            print("[OK] benchmark_v2 pre-flight checks passed (legacy v1 checksum skipped).")

    dataset_checksum = sha256_file(data_path)
    print(f"Dataset checksum: {dataset_checksum}")

    df = pd.read_csv(data_path)
    
    # Validate labels
    invalid_labels = set(df["error_label"]) - set(ERROR_LABELS)
    if invalid_labels:
        print(f"ERROR: Invalid labels found: {invalid_labels}")
        return
    
    # Check for missing data
    if df["code"].isna().any():
        print("WARNING: Removing samples with missing code.")
        df = df.dropna(subset=["code"])
    
    df["label_id"] = df["error_label"].map(LABEL_TO_ID)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    print(f"\nClass distribution (train):")
    print(train_df["error_label"].value_counts().to_string())
    
    if len(train_df) == 0:
        print("ERROR: No training samples.")
        return
    
    # Load tokenizer and model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ERROR_LABELS),
    )
    model.to(device)
    
    input_mode = getattr(args, "input_mode", "code_only")
    from training.spec_code_input import build_texts_from_frame

    train_texts = build_texts_from_frame(train_df, input_mode)
    val_texts = build_texts_from_frame(val_df, input_mode) if len(val_df) > 0 else []

    train_dataset = CodeDataset(
        train_texts,
        train_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )
    val_dataset = CodeDataset(
        val_texts,
        val_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_length,
    ) if len(val_df) > 0 else None
    
    train_sampler = None
    shuffle = True
    if getattr(args, "balanced_sampler", False):
        from torch.utils.data import WeightedRandomSampler

        train_labels = train_df["label_id"].tolist()
        class_sample_count = np.bincount(train_labels, minlength=len(ERROR_LABELS))
        weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
        sample_weights = [float(weights_per_class[l]) for l in train_labels]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    ) if val_dataset else None
    
    class_counts = train_df["label_id"].value_counts().sort_index().values
    total = sum(class_counts)
    use_class_weights = not getattr(args, "no_class_weights", False)
    if use_class_weights:
        weights = torch.tensor(
            [total / (len(class_counts) * c) for c in class_counts],
            dtype=torch.float32,
        ).to(device)
    else:
        weights = None
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = -1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    patience = getattr(args, "patience", 0)
    epochs_without_improve = 0
    
    training_history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max length: {args.max_length}")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        epoch_time = time.time() - start_time
        
        # Validate
        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_model(
                model, val_loader, device, criterion=criterion, detailed=True, return_probabilities=True
            )
            val_f1 = val_metrics["macro_f1"]
            val_loss = val_metrics.get("loss", 0.0)
            preds = np.argmax(val_metrics.get("probabilities") or [], axis=1) if val_metrics.get("probabilities") else []
            pred_dist = Counter(ID_TO_LABEL[int(p)] for p in preds).most_common() if len(preds) else []
            max_frac = (pred_dist[0][1] / len(preds)) if pred_dist and len(preds) else 0

            print(f"\n  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_f1:.4f}")
            if max_frac >= 0.75 and pred_dist:
                print(f"    [WARN] Possible class collapse: {pred_dist[0][0]} = {max_frac:.0%} of val preds")

            diag = {
                "epoch": epoch + 1,
                "predicted_label_distribution": dict(pred_dist),
                "max_predicted_class_fraction": round(max_frac, 4),
                "collapse_warning": max_frac >= 0.75,
                "per_class_recall": {lbl: val_metrics.get("per_class", {}).get(lbl, {}).get("recall") for lbl in ERROR_LABELS},
                "confusion_matrix": val_metrics.get("confusion_matrix").tolist() if hasattr(val_metrics.get("confusion_matrix"), "tolist") else None,
            }
            with open(metrics_dir / f"validation_diagnostics_epoch{epoch+1}.json", "w") as vf:
                json.dump(diag, vf, indent=2)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improve = 0
                model.save_pretrained(str(output_dir))
                tokenizer.save_pretrained(str(output_dir))
                print(f"    [OK] New best model saved (F1: {val_f1:.4f})")
            else:
                epochs_without_improve += 1
            if patience > 0 and epochs_without_improve >= patience:
                print(f"    Early stop: no val F1 improvement for {patience} epochs.")
                break
        else:
            # No validation set - save every epoch
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            print(f"\n  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        history_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
        }
        for k, v in val_metrics.items():
            if k in ("confusion_matrix", "report_dict", "report_text", "probabilities", "per_class"):
                continue
            history_row[f"val_{k}"] = round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
        training_history.append(history_row)
    
    # Save best-checkpoint validation metrics (always)
    if val_loader and best_epoch > 0:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(str(output_dir), num_labels=len(ERROR_LABELS))
        model.to(device)
        val_metrics_best = evaluate_model(
            model, val_loader, device, detailed=True, return_probabilities=True
        )
        preds_v = np.argmax(val_metrics_best["probabilities"], axis=1)
        from sklearn.metrics import f1_score as sk_f1

        y_val = val_df["label_id"].values
        weighted_f1 = float(sk_f1(y_val, preds_v, average="weighted", zero_division=0))
        fault_mask = y_val != LABEL_TO_ID["correct"]
        fault_recall = (
            float(sk_f1(y_val[fault_mask], preds_v[fault_mask], average="macro", zero_division=0))
            if fault_mask.any()
            else None
        )
        fp_correct = int(((preds_v == LABEL_TO_ID["correct"]) & (y_val != LABEL_TO_ID["correct"])).sum())
        pred_counts = Counter(ID_TO_LABEL[int(p)] for p in preds_v)
        val_out = {
            "eval_split": "val",
            "dataset_version": (
                "benchmark_v3"
                if "benchmark_v3" in str(data_path) or "phase10" in str(data_path)
                else None
            ),
            "best_epoch": best_epoch,
            "macro_f1": round(val_metrics_best["macro_f1"], 4),
            "weighted_f1": round(weighted_f1, 4),
            "accuracy": round(val_metrics_best["accuracy"], 4),
            "per_class_metrics": val_metrics_best.get("per_class", {}),
            "predicted_label_counts": dict(pred_counts),
            "sample_count": len(val_df),
            "collapse_warning": max(pred_counts.values()) / max(len(preds_v), 1) >= 0.75,
            "fault_only_macro_f1_proxy": round(fault_recall, 4) if fault_recall is not None else None,
            "correct_code_false_positive_count": fp_correct,
            "label_quality": "synthetic_unreviewed",
        }
        with open(metrics_dir / "validation_metrics.json", "w") as f:
            json.dump(val_out, f, indent=2)
        val_preds_json = []
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            probs = {
                ID_TO_LABEL[j]: round(float(val_metrics_best["probabilities"][i][j]), 4)
                for j in range(len(ERROR_LABELS))
            }
            val_preds_json.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row.get("family_id"),
                    "true_label": row["error_label"],
                    "predicted_label": ID_TO_LABEL[int(preds_v[i])],
                    "correct": row["error_label"] == ID_TO_LABEL[int(preds_v[i])],
                    "probabilities": probs,
                }
            )
        with open(metrics_dir / "validation_predictions.json", "w") as f:
            json.dump(val_preds_json, f, indent=2)
        if val_metrics_best.get("confusion_matrix") is not None:
            with open(metrics_dir / "validation_confusion_matrix.json", "w") as f:
                json.dump(
                    {"labels": ERROR_LABELS, "matrix": val_metrics_best["confusion_matrix"].tolist()},
                    f,
                    indent=2,
                )

    skip_test = getattr(args, "skip_test_eval", False)
    if skip_test:
        print("\n[SKIP] Test evaluation disabled (ablation / validation-only mode).")
    elif len(test_df) > 0:
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set (historical benchmark)")
        print("=" * 60)
        # Reload best model
        if best_epoch > 0:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(output_dir), num_labels=len(ERROR_LABELS)
            )
            model.to(device)
        
        test_texts = build_texts_from_frame(test_df, input_mode)
        test_dataset = CodeDataset(
            test_texts,
            test_df["label_id"].tolist(),
            tokenizer,
            max_length=args.max_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        import time as _time

        _time.perf_counter()
        test_metrics = evaluate_model(
            model, test_loader, device, detailed=True, return_probabilities=True
        )
        infer_start = _time.perf_counter()
        _ = evaluate_model(model, test_loader, device, detailed=False)
        infer_elapsed = _time.perf_counter() - infer_start
        per_sample_ms = (infer_elapsed / len(test_df) * 1000) if len(test_df) else None
        
        print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test Macro F1:  {test_metrics['macro_f1']:.4f}")
        print(f"\nClassification Report:")
        print(test_metrics.get("report_text", ""))
        
        # Save test metrics
        metrics_dir = Path(args.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        prob_rows = test_metrics.get("probabilities") or []
        y_pred_test = np.argmax(prob_rows, axis=1) if len(prob_rows) else np.array([])
        weighted_f1 = float(
            f1_score(test_df["label_id"].values, y_pred_test, average="weighted", zero_division=0)
        ) if len(y_pred_test) else 0.0

        test_output = {
            "model_type": "codebert_finetuned",
            "dataset_version": (
                "benchmark_v3"
                if "benchmark_v3" in str(data_path) or "phase10" in str(data_path)
                else ("benchmark_v2" if "benchmark_v2" in str(data_path) else "legacy_v1")
            ),
            "eval_split": "test",
            "accuracy": round(test_metrics["accuracy"], 4),
            "macro_precision": round(test_metrics["macro_precision"], 4),
            "macro_recall": round(test_metrics["macro_recall"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class_metrics": test_metrics.get("per_class", {}),
            "sample_count": len(test_df),
            "test_family_count": int(test_df["family_id"].nunique()) if "family_id" in test_df.columns else None,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_f1, 4),
            "dataset_checksum_sha256": dataset_checksum,
            "inference_ms_per_sample_test_cpu": round(per_sample_ms, 2) if per_sample_ms else None,
            "probability_note": "Softmax scores — not calibrated confidence.",
            "evaluated_at": pd.Timestamp.now().isoformat(),
        }
        
        with open(metrics_dir / "codebert_classification_metrics.json", "w") as f:
            json.dump(test_output, f, indent=2)

        if "report_dict" in test_metrics:
            with open(metrics_dir / "codebert_classification_report.json", "w") as f:
                json.dump(test_metrics["report_dict"], f, indent=2)

        cb_predictions = []
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            probs = None
            if prob_rows:
                probs = {
                    ID_TO_LABEL[j]: round(float(prob_rows[i][j]), 4)
                    for j in range(len(ERROR_LABELS))
                }
            cb_predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row.get("family_id"),
                    "true_label": row["error_label"],
                    "predicted_id": int(y_pred_test[i]) if len(y_pred_test) else None,
                    "predicted_label": ID_TO_LABEL[int(y_pred_test[i])] if len(y_pred_test) else None,
                    "probabilities": probs,
                }
            )
        with open(metrics_dir / "codebert_test_predictions.json", "w") as f:
            json.dump(cb_predictions, f, indent=2)

        baseline_path = metrics_dir.parent / "20261019_062841" / "classification_metrics.json"
        if not baseline_path.exists():
            baseline_path = PROJECT_ROOT / "artifacts" / "metrics" / "baseline_classification_metrics.json"
        if baseline_path.exists():
            with open(baseline_path) as bf:
                baseline = json.load(bf)
            comparison = {
                "test_sample_count": len(test_df),
                "dataset_checksum_sha256": dataset_checksum,
                "codebert_test_macro_f1": test_output["macro_f1"],
                "baseline_test_macro_f1": baseline.get("macro_f1"),
                "codebert_beats_baseline_macro_f1": (
                    test_output["macro_f1"] > baseline.get("macro_f1", 0)
                ),
            }
            with open(metrics_dir / "classification_model_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)

        with open(metrics_dir / "eval_config.json", "w") as f:
            json.dump(
                {
                    "checkpoint_dir": str(output_dir),
                    "metrics_dir": str(metrics_dir),
                    "dataset_checksum_sha256": dataset_checksum,
                    "checkpoint_selection": "validation_macro_f1",
                    "production_checkpoint_touched": False,
                },
                f,
                indent=2,
            )
        with open(metrics_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        # Save confusion matrix
        if "confusion_matrix" in test_metrics:
            cm_data = {
                "labels": ERROR_LABELS,
                "matrix": test_metrics["confusion_matrix"].tolist(),
                "model_type": "codebert",
            }
            with open(metrics_dir / "confusion_matrix.json", "w") as f:
                json.dump(cm_data, f, indent=2)

        if "benchmark_v2" in str(data_path):
            try:
                import subprocess as _sp

                _sp.check_call(
                    [
                        sys.executable,
                        "-m",
                        "training.classification_error_gallery",
                        "--codebert-run",
                        metrics_dir.name,
                    ],
                    cwd=str(PROJECT_ROOT),
                )
            except Exception as exc:
                print(f"[WARN] Error gallery generation skipped: {exc}")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save label mapping
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
            "labels": ERROR_LABELS,
        }, f, indent=2)
    
    # Save training config
    config = {
        "model_name": args.model_name,
        "num_labels": len(ERROR_LABELS),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "dataset_checksum_sha256": dataset_checksum,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "checkpoint_selection": "validation_macro_f1",
        "class_weights": use_class_weights,
        "balanced_sampler": getattr(args, "balanced_sampler", False),
        "experiment_id": getattr(args, "experiment_id", None),
        "skip_test_eval": getattr(args, "skip_test_eval", False),
        "patience": getattr(args, "patience", 0),
        "input_mode": input_mode,
    }
    with open(metrics_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Training complete!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val F1: {best_val_f1:.4f}")
    print(f"   Model saved to: {output_dir}")


def evaluate_model(model, dataloader, device, detailed=False, criterion=None, return_probabilities=False):
    """Evaluate the model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    loss_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if criterion is not None:
                total_loss += criterion(outputs.logits, labels).item()
                loss_batches += 1
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            if return_probabilities:
                probs = torch.softmax(logits, dim=-1)
                all_probabilities.extend(probs.cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    if loss_batches > 0:
        metrics["loss"] = total_loss / loss_batches
    
    if detailed:
        labels_idx = list(range(len(ERROR_LABELS)))
        report = classification_report(
            all_labels,
            all_preds,
            labels=labels_idx,
            target_names=ERROR_LABELS,
            output_dict=True,
            zero_division=0,
        )
        metrics["report_dict"] = report
        metrics["report_text"] = classification_report(
            all_labels,
            all_preds,
            labels=labels_idx,
            target_names=ERROR_LABELS,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds, labels=labels_idx)
        
        per_class = {}
        for label in ERROR_LABELS:
            if label in report:
                per_class[label] = {
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
        metrics["per_class"] = per_class
        if return_probabilities:
            metrics["probabilities"] = all_probabilities
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for error classification")
    parser.add_argument("--data_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "classification" / "dataset.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "codebert_classifier"))
    parser.add_argument("--metrics_dir", type=str,
                        default=str(PROJECT_ROOT / "artifacts" / "metrics"))
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_preflight",
        action="store_true",
        help="Skip all dataset pre-flight checks (not recommended)",
    )
    parser.add_argument("--no_class_weights", action="store_true", help="Use unweighted cross-entropy")
    parser.add_argument("--balanced_sampler", action="store_true", help="WeightedRandomSampler on train labels")
    parser.add_argument("--skip_test_eval", action="store_true", help="Validation-only; do not evaluate test split")
    parser.add_argument("--experiment_id", type=str, default=None, help="Ablation experiment label")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping epochs without val F1 gain (0=off)")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="code_only",
        choices=("code_only", "spec_code"),
        help="code_only or spec_code ([TASK] problem_type + [CODE] source)",
    )

    args = parser.parse_args()
    train(args)
