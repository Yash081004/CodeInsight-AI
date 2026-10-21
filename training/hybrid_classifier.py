"""
Hybrid classifier: TF-IDF + LogReg with deterministic static overrides (separate system).

Static rules only fire when the model margin is low and a specific heuristic matches.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.app.config import ERROR_LABELS, ID_TO_LABEL, LABEL_TO_ID


def static_label_hint(code: str) -> Optional[str]:
    """Return a suggested error label from syntax/AST heuristics, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    if re.search(r"product\s*=\s*0|acc\s*=\s*0.*\*", code):
        if re.search(r"\*=|product|factorial", code, re.I):
            return "wrong_initialization"

    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.BinOp):
                    return "off_by_one_loop"
                if len(node.iter.args) >= 2:
                    return "off_by_one_loop"

    if re.search(r"if\s+[^:]+\s*[<>=!]=", code) and re.search(r"return\s+\w+", code):
        if re.search(r"max|min|greater|less", code, re.I):
            return "wrong_condition"

    if re.search(r"\+\s*=\s*[^+]|\-\s*=\s*[^-]|\*\s*=\s*[^*]|/\s*=\s*[^/]", code):
        if re.search(r"sum|total|acc", code, re.I) and re.search(r"\-\s*=", code):
            return "wrong_operator"

    return None


def predict_hybrid(code: str, clf, vec, margin_threshold: float = 0.22) -> tuple[int, str]:
    X = vec.transform([code])
    proba = clf.predict_proba(X)[0]
    pred_id = int(proba.argmax())
    margin = float(proba.max() - sorted(proba)[-2]) if len(proba) > 1 else float(proba.max())

    hint = static_label_hint(code)
    if hint and margin < margin_threshold and hint in LABEL_TO_ID:
        return LABEL_TO_ID[hint], "static_override"
    return pred_id, "model"


def evaluate_hybrid_on_test(
    dataset_csv: Path,
    clf_path: Path,
    vec_path: Path,
    margin_threshold: float = 0.22,
) -> dict:
    df = pd.read_csv(dataset_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    y_true = test_df["label_id"].tolist() if "label_id" in test_df.columns else test_df["error_label"].map(LABEL_TO_ID).tolist()
    y_pred = []
    overrides = 0
    for code in test_df["code"]:
        pid, src = predict_hybrid(code, clf, vec, margin_threshold)
        if src == "static_override":
            overrides += 1
        y_pred.append(pid)

    return {
        "model": "hybrid_tfidf_static",
        "eval_split": "test",
        "test_n": len(test_df),
        "static_overrides": overrides,
        "margin_threshold": margin_threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "note": "Separate system; static rules are heuristics not ground truth.",
    }
