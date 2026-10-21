"""One-off diagnostic: production classifier logits on five canonical cases."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, get_settings

CASES = [
    ("A", "correct", "def square(n):\n    return n * n", None),
    (
        "B",
        "wrong_operator",
        "def square(n):\n    return n + n",
        "Return the square of the given number.",
    ),
    (
        "C",
        "off_by_one_loop",
        "def sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total",
        "Return the sum of all integers from 1 through n, inclusive.",
    ),
    (
        "D",
        "wrong_condition",
        "def is_adult(age):\n    if age < 18:\n        return True\n    return False",
        "Return True if age is at least 18, otherwise return False.",
    ),
    (
        "E",
        "wrong_initialization",
        "def sum_values(values):\n    total = 1\n    for value in values:\n        total += value\n    return total",
        "Return the sum of all elements in the list. The sum of an empty list should be zero.",
    ),
]


def run_codebert(model_path: Path, code: str, max_length: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=len(ERROR_LABELS),
    )
    model.eval()
    inputs = tokenizer(
        code,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy().flatten()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_id = int(np.argmax(probs))
    return {
        "logits": {ID_TO_LABEL[i]: float(logits[i]) for i in range(len(ERROR_LABELS))},
        "probs": {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(ERROR_LABELS))},
        "predicted": ID_TO_LABEL[pred_id],
        "confidence": float(probs[pred_id]),
        "config_id2label": model.config.id2label,
        "max_length_used": max_length,
    }


def main():
    settings = get_settings()
    ckpt = settings.classifier_model_path.resolve()
    print("checkpoint:", ckpt)
    print("exists:", ckpt.exists(), "safetensors:", (ckpt / "model.safetensors").exists())
    train_cfg = json.loads((ckpt / "training_config.json").read_text()) if (ckpt / "training_config.json").exists() else {}
    train_max = train_cfg.get("max_length", 128)
    print("training max_length:", train_max)

    for case_id, expected, code, problem in CASES:
        r512 = run_codebert(ckpt, code, max_length=512)
        r128 = run_codebert(ckpt, code, max_length=train_max)
        print("\n=== Case", case_id, "expected", expected, "===")
        if problem:
            print("problem_statement (NOT fed to model in current API):", problem[:60], "...")
        print("pred@512:", r512["predicted"], r512["confidence"])
        print("pred@train_max:", r128["predicted"], r128["confidence"])
        print("logits@512:", json.dumps(r512["logits"], indent=2))
        print("probs@512:", json.dumps({k: round(v, 4) for k, v in r512["probs"].items()}))


if __name__ == "__main__":
    main()
