import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.config import ERROR_LABELS, get_settings

client = TestClient(app)


@pytest.fixture
def sample_v3_experiment(tmp_path, monkeypatch):
    """Inject experiment artifacts under an isolated artifacts root."""
    exp_id = "gpu_1_baseline"
    artifacts_root = tmp_path / "artifacts"
    root = artifacts_root / "metrics" / "benchmark_v3" / "codebert_experiments"
    exp_dir = root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    vm = {
        "eval_split": "val",
        "macro_f1": 0.12,
        "accuracy": 0.5,
        "collapse_warning": True,
        "predicted_label_counts": {"correct": 25, "wrong_operator": 5},
        "per_class_metrics": {
            lbl: {"precision": 0.1, "recall": 0.1, "f1": 0.1, "support": 6} for lbl in ERROR_LABELS
        },
        "label_quality": "synthetic_unreviewed",
    }
    cm = {"labels": ERROR_LABELS, "matrix": [[1] * len(ERROR_LABELS) for _ in ERROR_LABELS]}
    (exp_dir / "validation_metrics.json").write_text(json.dumps(vm), encoding="utf-8")
    (exp_dir / "validation_confusion_matrix.json").write_text(json.dumps(cm), encoding="utf-8")
    (exp_dir / "experiment_run.json").write_text(
        json.dumps({"experiment_id": exp_id, "gpu_run": False, "device_used": "cpu"}),
        encoding="utf-8",
    )
    summary = {
        "selection": {
            "selected_experiment_id": exp_id,
            "validation_macro_f1": 0.12,
            "beats_tfidf_val_macro_f1": False,
        }
    }
    (root / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    settings = get_settings()
    mock_settings = MagicMock()
    mock_settings.artifacts_path = artifacts_root
    monkeypatch.setattr("backend.app.api.evaluation.get_settings", lambda: mock_settings)
    yield exp_id


def test_v3_codebert_list_and_detail(sample_v3_experiment):
    exp_id = sample_v3_experiment
    r = client.get("/api/evaluation/benchmark-v3-codebert")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is True
    assert data["dataset_version"] == "benchmark_v3"
    assert data["label_quality_warning"]
    ids = [e["experiment_id"] for e in data["experiments"]]
    assert exp_id in ids

    r2 = client.get(f"/api/evaluation/benchmark-v3-codebert?experiment_id={exp_id}")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["available"] is True
    assert d2["validation"]["macro_f1"] == 0.12
    assert d2["confusion_matrix"]["labels"] == ERROR_LABELS


def test_v3_codebert_rejects_invalid_experiment_id():
    r = client.get("/api/evaluation/benchmark-v3-codebert?experiment_id=../../../etc/passwd")
    assert r.status_code == 200
    assert r.json()["available"] is False

    r2 = client.get("/api/evaluation/benchmark-v3-codebert?experiment_id=unknown_run")
    assert r2.json()["available"] is False


def test_collapse_detection_threshold():
    from training.benchmark_runner import collapsed_predictions
    from backend.app.config import LABEL_TO_ID
    import numpy as np

    preds = np.array([LABEL_TO_ID["correct"]] * 23 + [LABEL_TO_ID["wrong_operator"]] * 7)
    c = collapsed_predictions(preds)
    assert c["collapsed"] is True
    assert c["dominant_class"] == "correct"
