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
    """PyTorch dataset for tokenized code samples."""
    
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
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
    
    # Datasets
    train_dataset = CodeDataset(
        train_df["code"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )
    val_dataset = CodeDataset(
        val_df["code"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_length,
    ) if len(val_df) > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    ) if val_dataset else None
    
    # Compute class weights for imbalanced data
    class_counts = train_df["label_id"].value_counts().sort_index().values
    total = sum(class_counts)
    weights = torch.tensor(
        [total / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Loss with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = -1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            val_metrics = evaluate_model(model, val_loader, device)
            val_f1 = val_metrics["macro_f1"]
            
            print(f"\n  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"    Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_f1:.4f}")
            
            # Save best checkpoint based on validation macro F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                
                model.save_pretrained(str(output_dir))
                tokenizer.save_pretrained(str(output_dir))
                
                print(f"    [OK] New best model saved (F1: {val_f1:.4f})")
        else:
            # No validation set - save every epoch
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            print(f"\n  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        })
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    if len(test_df) > 0:
        # Reload best model
        if best_epoch > 0:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(output_dir), num_labels=len(ERROR_LABELS)
            )
            model.to(device)
        
        test_dataset = CodeDataset(
            test_df["code"].tolist(),
            test_df["label_id"].tolist(),
            tokenizer,
            max_length=args.max_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        test_metrics = evaluate_model(model, test_loader, device, detailed=True)
        
        print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test Macro F1:  {test_metrics['macro_f1']:.4f}")
        print(f"\nClassification Report:")
        print(test_metrics.get("report_text", ""))
        
        # Save test metrics
        metrics_dir = Path(args.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        test_output = {
            "model_type": "codebert_finetuned",
            "accuracy": round(test_metrics["accuracy"], 4),
            "macro_precision": round(test_metrics["macro_precision"], 4),
            "macro_recall": round(test_metrics["macro_recall"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "per_class_metrics": test_metrics.get("per_class", {}),
            "sample_count": len(test_df),
            "best_epoch": best_epoch,
            "evaluated_at": pd.Timestamp.now().isoformat(),
        }
        
        with open(metrics_dir / "codebert_classification_metrics.json", "w") as f:
            json.dump(test_output, f, indent=2)
        
        # Save confusion matrix
        if "confusion_matrix" in test_metrics:
            cm_data = {
                "labels": ERROR_LABELS,
                "matrix": test_metrics["confusion_matrix"].tolist(),
                "model_type": "codebert",
            }
            with open(metrics_dir / "confusion_matrix.json", "w") as f:
                json.dump(cm_data, f, indent=2)
    
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Training complete!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val F1: {best_val_f1:.4f}")
    print(f"   Model saved to: {output_dir}")


def evaluate_model(model, dataloader, device, detailed=False):
    """Evaluate the model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
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
    
    if detailed:
        report = classification_report(
            all_labels, all_preds,
            target_names=ERROR_LABELS,
            output_dict=True,
            zero_division=0,
        )
        metrics["report_text"] = classification_report(
            all_labels, all_preds,
            target_names=ERROR_LABELS,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
        
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
    
    args = parser.parse_args()
    train(args)
