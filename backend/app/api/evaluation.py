"""
CodeInsight AI - Evaluation API

Endpoints for model evaluation results and experiment runs.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from backend.app.schemas import EvaluationSummary, EvaluationMetrics
from backend.app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

PRODUCTION_RUN_ID = "production"
_BENCHMARK_RUN_RE = __import__("re").compile(r"^benchmark_v2/[A-Za-z0-9_\-]+$")
_V3_CODEBERT_EXP_RE = __import__("re").compile(r"^[a-z0-9_]+$")
_V3_CODEBERT_EXP_IDS = frozenset({"gpu_1_baseline", "gpu_2_lr3e5", "gpu_3_balanced"})


def _benchmark_subdir(run_id: str) -> Optional[str]:
    if not _BENCHMARK_RUN_RE.match(run_id):
        return None
    return run_id.split("/", 1)[1]


def _benchmark_metrics_dir(metrics_root: Path, run_id: str) -> Optional[Path]:
    sub = _benchmark_subdir(run_id)
    if not sub:
        return None
    path = metrics_root / "benchmark_v2" / sub
    return path if path.is_dir() else None


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return {}


def _metrics_from_file(path: Path) -> Optional[EvaluationMetrics]:
    data = _load_json(path)
    if not data:
        return None
    try:
        return EvaluationMetrics(
            model_type=data.get("model_type") or data.get("model", "unknown"),
            accuracy=float(data["accuracy"]),
            macro_precision=float(data["macro_precision"]),
            macro_recall=float(data["macro_recall"]),
            macro_f1=float(data["macro_f1"]),
            per_class_metrics=data.get("per_class_metrics", {}),
            sample_count=int(data.get("sample_count") or data.get("test_n") or 0),
            evaluated_at=data.get("evaluated_at"),
        )
    except Exception as e:
        logger.warning(f"Invalid metrics at {path}: {e}")
        return None


def _discover_benchmark_v2_runs(metrics_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    bench_dir = metrics_root / "benchmark_v2"
    if not bench_dir.exists():
        return runs
    for sub in sorted(bench_dir.iterdir()):
        if not sub.is_dir():
            continue
        mpath = sub / "classification_metrics.json"
        codebert_path = sub / "codebert_classification_metrics.json"
        metrics_file = codebert_path if codebert_path.exists() else mpath
        if not metrics_file.exists():
            continue
        data = _load_json(metrics_file)
        runs.append(
            {
                "run_id": f"benchmark_v2/{sub.name}",
                "role": "benchmark_v2",
                "checkpoint_path": f"models/benchmark_v2/{sub.name}"
                if (metrics_root.parent.parent / "models" / "benchmark_v2" / sub.name).exists()
                else None,
                "metrics_path": str(metrics_file.relative_to(metrics_root.parent.parent))
                if metrics_root.parent.parent.exists()
                else str(metrics_file),
                "dataset_version": data.get("dataset_version", "benchmark_v2"),
                "dataset_checksum_sha256": data.get("dataset_checksum_sha256") or data.get("checksum"),
                "eval_split": data.get("eval_split", "test"),
                "sample_count": data.get("sample_count") or data.get("test_n"),
                "macro_f1": data.get("macro_f1"),
                "evaluated_at": data.get("evaluated_at"),
                "served_by_api": False,
            }
        )
    return runs


def _discover_experiment_runs(metrics_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    experiments_dir = metrics_root / "experiments"
    if not experiments_dir.exists():
        return runs
    for sub in sorted(experiments_dir.iterdir()):
        if not sub.is_dir():
            continue
        mpath = sub / "codebert_classification_metrics.json"
        if not mpath.exists():
            continue
        data = _load_json(mpath)
        runs.append(
            {
                "run_id": sub.name,
                "role": "experiment",
                "checkpoint_path": f"models/experiments/{sub.name}",
                "metrics_path": str(mpath.relative_to(metrics_root.parent.parent)) if metrics_root.parent.parent.exists() else str(mpath),
                "dataset_checksum_sha256": data.get("dataset_checksum_sha256"),
                "eval_split": data.get("eval_split", "test"),
                "sample_count": data.get("sample_count"),
                "macro_f1": data.get("macro_f1"),
                "best_epoch": data.get("best_epoch"),
                "evaluated_at": data.get("evaluated_at"),
            }
        )
    return runs


@router.get("/experiments")
async def list_experiment_runs():
    """List production and experiment evaluation runs (metadata only)."""
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"

    production_metrics = _load_json(metrics_dir / "codebert_classification_metrics.json")
    baseline_metrics = _load_json(metrics_dir / "baseline_classification_metrics.json")
    sim_metrics = _load_json(metrics_dir / "similarity_metrics.json")

    runs: List[Dict[str, Any]] = [
        {
            "run_id": PRODUCTION_RUN_ID,
            "role": "production",
            "checkpoint_path": str(settings.classifier_model_path.relative_to(settings.models_path.parent))
            if settings.classifier_model_path.exists()
            else "models/codebert_classifier",
            "metrics_path": "artifacts/metrics/codebert_classification_metrics.json",
            "dataset_checksum_sha256": production_metrics.get("dataset_checksum_sha256"),
            "eval_split": production_metrics.get("eval_split", "test"),
            "sample_count": production_metrics.get("sample_count"),
            "macro_f1": production_metrics.get("macro_f1"),
            "evaluated_at": production_metrics.get("evaluated_at"),
            "served_by_api": True,
        }
    ]
    runs.extend(_discover_experiment_runs(metrics_dir))
    runs.extend(_discover_benchmark_v2_runs(metrics_dir))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_run_id": PRODUCTION_RUN_ID,
        "baseline_test_macro_f1": baseline_metrics.get("macro_f1"),
        "baseline_sample_count": baseline_metrics.get("sample_count"),
        "similarity_test_pairs": sim_metrics.get("test_pairs") or sim_metrics.get("tfidf_test_sample_count"),
        "similarity_preliminary": True,
        "runs": runs,
    }


@router.get("/summary", response_model=EvaluationSummary)
async def evaluation_summary(
    run_id: str = Query(PRODUCTION_RUN_ID, description="production or experiment folder name"),
):
    """Get evaluation results for a specific run (default: production metrics on disk)."""
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"

    if run_id == PRODUCTION_RUN_ID:
        codebert_path = metrics_dir / "codebert_classification_metrics.json"
        cm_path = metrics_dir / "confusion_matrix.json"
        baseline_path = metrics_dir / "baseline_classification_metrics.json"
        sim_path = metrics_dir / "similarity_metrics.json"
    elif run_id.startswith("benchmark_v2/"):
        bench = _benchmark_metrics_dir(metrics_dir, run_id)
        if not bench:
            bench = metrics_dir / "benchmark_v2" / run_id.split("/", 1)[1]
        codebert_path = bench / "codebert_classification_metrics.json"
        cm_path = bench / "confusion_matrix.json"
        sim_path = bench / "similarity_metrics.json"
        comparison = _load_json(bench / "model_comparison.json")
        bl_run = (comparison or {}).get("baseline_run_id", "20261019_062841")
        baseline_path = metrics_dir / "benchmark_v2" / bl_run / "classification_metrics.json"
        if not codebert_path.exists():
            codebert_path = bench / "classification_metrics.json"
            baseline_path = bench / "classification_metrics.json"
    else:
        codebert_path = metrics_dir / "experiments" / run_id / "codebert_classification_metrics.json"
        cm_path = metrics_dir / "experiments" / run_id / "confusion_matrix.json"
        baseline_path = metrics_dir / "baseline_classification_metrics.json"
        sim_path = metrics_dir / "similarity_metrics.json"

    result = EvaluationSummary(
        classification_available=False,
        similarity_available=False,
    )

    codebert_data = _load_json(codebert_path)
    if codebert_data:
        try:
            result.codebert_metrics = EvaluationMetrics(**codebert_data)
            result.classification_available = True
        except Exception as e:
            logger.warning(f"Invalid CodeBERT metrics format: {e}")

    baseline_data = _load_json(baseline_path)
    if baseline_data:
        try:
            result.baseline_metrics = EvaluationMetrics(**baseline_data)
            result.classification_available = True
        except Exception as e:
            logger.warning(f"Invalid baseline metrics format: {e}")

    sim_data = _load_json(sim_path)
    if sim_data:
        result.similarity_metrics = sim_data
        result.similarity_available = True

    cm_data = _load_json(cm_path)
    if cm_data:
        result.confusion_matrix = cm_data

    return result


@router.get("/confusion-matrix")
async def confusion_matrix(
    run_id: str = Query(PRODUCTION_RUN_ID),
    model: str = Query("codebert", description="codebert or baseline"),
):
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"

    if model == "baseline":
        cm_path = metrics_dir / "baseline_confusion_matrix.json"
    elif run_id == PRODUCTION_RUN_ID:
        cm_path = metrics_dir / "confusion_matrix.json"
    elif run_id.startswith("benchmark_v2/"):
        sub = run_id.split("/", 1)[1]
        cm_path = metrics_dir / "benchmark_v2" / sub / "confusion_matrix.json"
    else:
        cm_path = metrics_dir / "experiments" / run_id / "confusion_matrix.json"

    data = _load_json(cm_path)
    if not data:
        return {
            "available": False,
            "run_id": run_id,
            "model": model,
            "message": "No confusion matrix for this run/model combination.",
        }

    return {"available": True, "run_id": run_id, "model": model, "data": data}


@router.get("/error-gallery")
async def error_gallery(
    run_id: str = Query(..., description="benchmark_v2/<run_name> with gallery artifacts"),
    correct: Optional[bool] = Query(None),
    true_label: Optional[str] = Query(None),
    predicted_label: Optional[str] = Query(None),
    model: Optional[str] = Query(None, description="baseline or codebert filter on wrong predictions"),
    family_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"
    bench = _benchmark_metrics_dir(metrics_dir, run_id)
    if not bench:
        return {
            "available": False,
            "run_id": run_id,
            "message": "Invalid or missing benchmark run.",
        }
    gallery_path = bench / "classification_error_gallery.json"
    data = _load_json(gallery_path)
    if not data:
        return {
            "available": False,
            "run_id": run_id,
            "message": "Error gallery not generated for this run.",
        }
    entries = data.get("entries") or []
    filtered = []
    for e in entries:
        if true_label and e.get("true_label") != true_label:
            continue
        if family_id and e.get("family_id") != family_id:
            continue
        if correct is True and not (e["baseline"]["correct"] and e["codebert"]["correct"]):
            continue
        if correct is False and (e["baseline"]["correct"] and e["codebert"]["correct"]):
            continue
        if predicted_label:
            if model == "baseline":
                if e["baseline"]["predicted_label"] != predicted_label:
                    continue
            elif model == "codebert":
                if e["codebert"]["predicted_label"] != predicted_label:
                    continue
            elif (
                e["baseline"]["predicted_label"] != predicted_label
                and e["codebert"]["predicted_label"] != predicted_label
            ):
                continue
        if model == "baseline" and e["baseline"]["correct"]:
            continue
        if model == "codebert" and e["codebert"]["correct"]:
            continue
        filtered.append(e)
        if len(filtered) >= limit:
            break

    comparison = _load_json(bench / "model_comparison.json")
    return {
        "available": True,
        "run_id": run_id,
        "dataset_version": (comparison or {}).get("dataset_version", "benchmark_v2"),
        "matched_test_count": data.get("total"),
        "representative_keys": list((data.get("representative") or {}).keys()),
        "probability_note": "Softmax / logistic scores are not calibrated confidence.",
        "entries": filtered,
        "comparison_summary": {
            "baseline_macro_f1": (comparison or {}).get("baseline", {}).get("macro_f1"),
            "codebert_macro_f1": (comparison or {}).get("codebert", {}).get("macro_f1"),
            "baseline_run_id": (comparison or {}).get("baseline_run_id"),
            "codebert_run_id": (comparison or {}).get("codebert_run_id"),
        },
    }


@router.get("/benchmark-quality")
async def benchmark_quality(version: str = Query("benchmark_v3", pattern="^benchmark_v[23]$")):
    settings = get_settings()
    root = settings.artifacts_path.parent.parent / "data" / version
    report = _load_json(root / "quality_report.json")
    audit = _load_json(settings.artifacts_path / "phase5" / "audit_report.json")
    ds_audit = (audit.get("datasets") or {}).get(version, {}) if audit else {}
    if not report and not ds_audit:
        return {"available": False, "version": version}
    return {
        "available": True,
        "version": version,
        "quality_report": report,
        "integrity": ds_audit,
        "synthetic_unreviewed": report.get("synthetic_unreviewed", True),
        "human_reviewed": report.get("human_reviewed", False),
    }


def _v3_codebert_experiment_dir(settings, experiment_id: str) -> Optional[Path]:
    if experiment_id not in _V3_CODEBERT_EXP_IDS or not _V3_CODEBERT_EXP_RE.match(experiment_id):
        return None
    path = settings.artifacts_path / "metrics" / "benchmark_v3" / "codebert_experiments" / experiment_id
    return path if path.is_dir() else None


@router.get("/benchmark-v3-codebert")
async def benchmark_v3_codebert_experiments(
    experiment_id: Optional[str] = Query(None, description="Optional experiment id"),
):
    """Phase 6 benchmark_v3 CodeBERT matrix (validation-first; test optional post-selection)."""
    settings = get_settings()
    root = settings.artifacts_path / "metrics" / "benchmark_v3" / "codebert_experiments"
    summary = _load_json(root / "experiment_summary.json")
    preflight = _load_json(root / "preflight.json")
    tfidf = _load_json(settings.artifacts_path / "metrics" / "benchmark_v3" / "runner" / "tfidf_val_metrics.json")

    if experiment_id:
        exp_dir = _v3_codebert_experiment_dir(settings, experiment_id)
        if not exp_dir:
            return {"available": False, "message": "Invalid or missing experiment_id"}
        vm = _load_json(exp_dir / "validation_metrics.json")
        cm = _load_json(exp_dir / "validation_confusion_matrix.json")
        run_meta = _load_json(exp_dir / "experiment_run.json")
        final_test = _load_json(exp_dir / "final_test_metrics.json")
        return {
            "available": True,
            "experiment_id": experiment_id,
            "dataset_version": "benchmark_v3",
            "synthetic_unreviewed": True,
            "human_reviewed": False,
            "validation": vm,
            "confusion_matrix": cm,
            "run_meta": run_meta,
            "final_test": final_test or None,
            "tfidf_baseline_val_macro_f1": (tfidf or {}).get("macro_f1"),
        }

    experiments = []
    if root.exists():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or sub.name not in _V3_CODEBERT_EXP_IDS:
                continue
            vm = _load_json(sub / "validation_metrics.json")
            run_meta = _load_json(sub / "experiment_run.json")
            final_test = _load_json(sub / "final_test_metrics.json")
            experiments.append(
                {
                    "experiment_id": sub.name,
                    "gpu_run": (run_meta or {}).get("gpu_run"),
                    "device_used": (run_meta or {}).get("device_used"),
                    "validation_macro_f1": (vm or {}).get("macro_f1"),
                    "validation_accuracy": (vm or {}).get("accuracy"),
                    "validation_collapse_warning": (vm or {}).get("collapse_warning"),
                    "predicted_label_counts_val": (vm or {}).get("predicted_label_counts"),
                    "per_class_recall_val": {
                        lbl: (vm or {}).get("per_class_metrics", {}).get(lbl, {}).get("recall")
                        for lbl in [
                            "correct",
                            "wrong_initialization",
                            "off_by_one_loop",
                            "wrong_operator",
                            "wrong_condition",
                        ]
                    }
                    if vm
                    else None,
                    "final_test_macro_f1": (final_test or {}).get("macro_f1"),
                    "final_test_eval_split": (final_test or {}).get("eval_split"),
                }
            )
    return {
        "available": bool(experiments or summary),
        "dataset_version": "benchmark_v3",
        "label_quality_warning": "synthetic_unreviewed_not_human_verified",
        "preflight": preflight,
        "tfidf_baseline_val": tfidf,
        "selection": (summary or {}).get("selection"),
        "experiments": experiments,
        "policy_note": "Validation used for selection; test is optional single post-selection eval only.",
    }


@router.get("/ablations")
async def list_ablations():
    """benchmark_v2 CodeBERT ablation experiments (validation-first)."""
    settings = get_settings()
    ab_root = settings.artifacts_path / "metrics" / "benchmark_v2" / "ablations"
    if not ab_root.exists():
        return {"available": False, "experiments": []}
    summary = _load_json(ab_root / "ablation_summary.json")
    experiments = []
    for sub in sorted(ab_root.iterdir()):
        if not sub.is_dir():
            continue
        vm = _load_json(sub / "validation_metrics.json")
        cfg = _load_json(sub / "experiment_config.json")
        hist_test = _load_json(sub / "historical_test_metrics.json")
        experiments.append(
            {
                "experiment_id": sub.name,
                "description": (cfg or {}).get("experiment_id") or sub.name,
                "dataset_checksum_sha256": (cfg or {}).get("dataset_checksum_sha256"),
                "validation_macro_f1": (vm or {}).get("macro_f1"),
                "validation_collapse_warning": (vm or {}).get("collapse_warning"),
                "predicted_label_counts_val": (vm or {}).get("predicted_label_counts"),
                "per_class_recall_val": {
                    lbl: (vm or {}).get("per_class_metrics", {}).get(lbl, {}).get("recall")
                    for lbl in ["correct", "wrong_initialization", "off_by_one_loop", "wrong_operator", "wrong_condition"]
                }
                if vm
                else None,
                "historical_test_macro_f1": (hist_test or {}).get("macro_f1"),
                "historical_test_label": (hist_test or {}).get("label"),
                "training_config": cfg,
            }
        )
    return {
        "available": True,
        "dataset_version": "benchmark_v2",
        "selection_summary": summary,
        "experiments": experiments,
        "audit_path": "artifacts/metrics/benchmark_v2/phase4/collapse_audit.json",
    }


_PHASE7_MODELS = frozenset({"codebert", "tfidf"})
_PHASE7_SPLITS = frozenset({"val", "test"})


@router.get("/phase7-error-analysis")
async def phase7_error_analysis():
    settings = get_settings()
    report = _load_json(settings.artifacts_path / "metrics" / "phase7" / "error_analysis_report.json")
    reviewed = _load_json(settings.artifacts_path.parent.parent / "data" / "benchmark_reviewed" / "manifest.json")
    if not report:
        return {"available": False, "message": "Run: py -3.13 -m training.phase7_error_analysis"}
    return {
        "available": True,
        "report": report,
        "label_quality": "synthetic_unreviewed",
        "reviewed_slice": reviewed,
        "task_contract_doc": "docs/PHASE7_ERROR_ANALYSIS.md",
    }


@router.get("/error-explorer")
async def error_explorer(
    split: str = Query("val", pattern="^(val|test)$"),
    model: str = Query("codebert", pattern="^(codebert|tfidf)$"),
    true_label: Optional[str] = Query(None),
    predicted_label: Optional[str] = Query(None),
    family_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    settings = get_settings()
    if model not in _PHASE7_MODELS or split not in _PHASE7_SPLITS:
        return {"available": False, "message": "Invalid model or split"}
    labels_ok = {
        "correct",
        "wrong_initialization",
        "off_by_one_loop",
        "wrong_operator",
        "wrong_condition",
    }
    if true_label and true_label not in labels_ok:
        return {"available": False, "message": "Invalid true_label"}
    if predicted_label and predicted_label not in labels_ok:
        return {"available": False, "message": "Invalid predicted_label"}
    if family_id and not __import__("re").match(r"^[a-z0-9_]+$", family_id):
        return {"available": False, "message": "Invalid family_id"}

    blob = _load_json(settings.artifacts_path / "metrics" / "phase7" / "error_explorer_entries.json")
    if not blob:
        return {"available": False, "message": "Phase 7 artifacts missing"}
    key = "codebert" if model == "codebert" else "tfidf"
    entries = (blob.get(split) or {}).get(key) or []
    filtered = []
    for e in entries:
        if true_label and e.get("true_label") != true_label:
            continue
        if predicted_label and e.get("predicted_label") != predicted_label:
            continue
        if family_id and e.get("family_id") != family_id:
            continue
        filtered.append(e)
    filtered = filtered[:limit]
    return {
        "available": True,
        "split": split,
        "model": model,
        "dataset_version": "benchmark_v3",
        "label_quality": "synthetic_unreviewed",
        "comparison_warning": (
            "CodeBERT rows use research checkpoint gpu_3_balanced; TF-IDF is re-fit on train only. "
            "Do not compare to production CodeBERT or benchmark_v2 metrics."
        ),
        "count": len(filtered),
        "entries": filtered,
    }


@router.get("/review-packet")
async def review_packet(
    review_id: Optional[str] = Query(None),
    sample_id: Optional[str] = Query(None),
    show_model_hints: bool = Query(False),
):
    from training.review_workflow import build_review_packet, load_decisions

    if not review_id and not sample_id:
        return {"available": False, "message": "Provide review_id or sample_id"}
    if sample_id and not __import__("re").match(r"^S[0-9]{4}$", sample_id):
        return {"available": False, "message": "Invalid sample_id"}
    if review_id and not __import__("re").match(r"^V3RQ[0-9]{3}$", review_id):
        return {"available": False, "message": "Invalid review_id"}

    data = load_decisions("benchmark_v3")
    target = None
    for d in data.get("decisions") or []:
        if review_id and d.get("review_id") == review_id:
            target = d
            break
        if sample_id and d.get("sample_id") == sample_id:
            target = d
            break
    if not target:
        return {"available": False, "message": "Review item not found"}
    pkt = build_review_packet(target, include_model_hints=show_model_hints)
    return {"available": True, "packet": pkt, "synthetic_unreviewed_until_decided": target.get("status") == "pending"}


@router.get("/review-queue")
async def review_queue(
    dataset: str = Query("benchmark_v3", pattern="^benchmark_v3$"),
    limit: int = Query(100, ge=1, le=200),
    priority_only: bool = Query(False),
):
    from backend.app.services.review_service import build_review_lab_summary

    if dataset != "benchmark_v3":
        return {"available": False, "message": "Only benchmark_v3 supported"}
    payload = build_review_lab_summary(limit=limit, priority_only=priority_only)
    payload["dataset_version"] = dataset
    return payload


@router.get("/review-lab")
async def review_lab(
    limit: int = Query(40, ge=1, le=200),
    priority_only: bool = Query(False),
):
    """Evaluation Lab: human review status (read-only; no decisions recorded)."""
    from backend.app.services.review_service import build_review_lab_summary

    return build_review_lab_summary(limit=limit, priority_only=priority_only)


@router.get("/model-comparison")
async def model_comparison(run_id: str = Query(...)):
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"
    bench = _benchmark_metrics_dir(metrics_dir, run_id)
    if not bench:
        return {"available": False, "run_id": run_id}
    data = _load_json(bench / "model_comparison.json")
    if not data:
        return {"available": False, "run_id": run_id, "message": "No model_comparison.json"}
    return {"available": True, "run_id": run_id, "data": data}
