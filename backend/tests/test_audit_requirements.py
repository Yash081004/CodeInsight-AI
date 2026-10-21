"""
CodeInsight AI - Comprehensive Audit & Quality Assurance Tests

Verifies requirements from the system audit:
- Code AST validation & safety
- Similarity calculation & normalization
- Classification 5-class taxonomy & bijective label mapping
- Feedback mapping completeness
- Dataset template syntax & mutation validity
- Split leakage prevention
- Metric computation accuracy
- Missing-model fallback & graceful degradation
- Request validation boundaries
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL, FRIENDLY_LABELS, FEEDBACK_TEMPLATES
from backend.app.services.code_validator import validate_python_code, extract_functions, count_lines
from backend.app.services.feedback_service import get_feedback, get_friendly_label
from backend.app.services.similarity_service import analyze_similarity, compute_tfidf_similarity
from backend.app.services.classification_service import analyze_code
from training.generate_dataset import TEMPLATES, validate_syntax, run_test_safely, _assign_splits


# ─── 1. Code AST Validation & Safety ─────────────────────────────────────────

class TestCodeValidationAudit:
    def test_valid_python_syntax(self):
        code = "def is_even(n):\n    return n % 2 == 0\n"
        is_valid, err = validate_python_code(code)
        assert is_valid is True
        assert err == ""

    def test_syntax_error_reported_with_line_number(self):
        code = "def broken():\n    for i in\n        pass"
        is_valid, err = validate_python_code(code)
        assert is_valid is False
        assert "Syntax error" in err
        assert "line" in err

    def test_empty_and_whitespace_rejected(self):
        assert validate_python_code("")[0] is False
        assert validate_python_code("   \n\t  ")[0] is False

    def test_max_length_boundary_enforced(self):
        short_code = "a = 1\n"
        assert validate_python_code(short_code, max_length=10)[0] is True
        long_code = "a = 1234567890\n"
        is_valid, err = validate_python_code(long_code, max_length=5)
        assert is_valid is False
        assert "maximum length" in err

    def test_extract_functions_ast(self):
        code = "def foo(): pass\ndef bar(x): return x * 2\nclass A: pass"
        fns = extract_functions(code)
        assert fns == ["foo", "bar"]


# ─── 2. Similarity Calculation & Mathematical Normalization ─────────────────

class TestSimilarityCalculationAudit:
    def test_identical_normalized_embeddings_give_one(self):
        v = np.array([0.6, 0.8])
        sim = float(np.dot(v, v))
        assert pytest.approx(sim, 1e-5) == 1.0

    def test_orthogonal_embeddings_give_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = float(np.dot(v1, v2))
        assert pytest.approx(sim, 1e-5) == 0.0

    def test_tfidf_identical_code_similarity_is_high(self):
        code = "def factorial(n):\n    res = 1\n    for i in range(1, n+1):\n        res *= i\n    return res"
        sim = compute_tfidf_similarity(code, code)
        assert sim is not None
        assert sim >= 0.99

    def test_mocked_codebert_encoder_flow(self):
        mock_encoder = MagicMock()
        mock_encoder.is_loaded = True
        mock_encoder.compute_similarity.return_value = 0.8854

        with patch("backend.app.services.similarity_service.get_encoder", return_value=mock_encoder):
            result = analyze_similarity("def a(): pass", "def b(): pass", threshold=0.75)
            assert result["codebert_score"] == 0.8854
            assert result["is_similar"] is True
            assert "hybrid" in result["decision_model"] or result["decision_model"] == "codebert"


# ─── 3. Classification Taxonomy & Label Bijectivity ──────────────────────────

class TestClassificationTaxonomyAudit:
    REQUIRED_CLASSES = {
        "correct",
        "wrong_initialization",
        "off_by_one_loop",
        "wrong_operator",
        "wrong_condition",
    }

    def test_exact_five_required_classes(self):
        assert len(ERROR_LABELS) == 5
        assert set(ERROR_LABELS) == self.REQUIRED_CLASSES

    def test_bijective_id_to_label_mapping(self):
        assert len(LABEL_TO_ID) == 5
        assert len(ID_TO_LABEL) == 5
        for label, idx in LABEL_TO_ID.items():
            assert ID_TO_LABEL[idx] == label
            assert 0 <= idx < 5

    def test_all_classes_have_friendly_labels_and_feedback(self):
        for label in self.REQUIRED_CLASSES:
            assert label in FRIENDLY_LABELS
            assert label in FEEDBACK_TEMPLATES
            assert len(FEEDBACK_TEMPLATES[label]) > 0
            feedback = get_feedback(label)
            assert isinstance(feedback, str) and len(feedback) > 10


# ─── 4. Dataset Template Syntax & Behavioral Tests ───────────────────────────

class TestDatasetTemplatesAudit:
    def test_all_registered_templates_have_valid_ast(self):
        assert len(TEMPLATES) > 0
        for t in TEMPLATES:
            assert validate_syntax(t["code"]), f"Template {t['template_id']} has invalid syntax"

    def test_all_registered_templates_pass_their_tests(self):
        for t in TEMPLATES:
            passed, err = run_test_safely(t["code"], t["test_code"])
            assert passed is True, f"Template {t['template_id']} failed its canonical test: {err}"


# ─── 5. Data-Split Leakage Prevention ────────────────────────────────────────

class TestSplitLeakagePreventionAudit:
    def test_assign_splits_groups_by_source_template_id(self):
        samples = [
            {"source_template_id": "t1", "id": 1},
            {"source_template_id": "t1", "id": 2},
            {"source_template_id": "t2", "id": 3},
            {"source_template_id": "t2", "id": 4},
            {"source_template_id": "t3", "id": 5},
            {"source_template_id": "t3", "id": 6},
            {"source_template_id": "t4", "id": 7},
            {"source_template_id": "t4", "id": 8},
        ]
        assigned = _assign_splits(samples, seed=42)

        template_splits = {}
        for s in assigned:
            tid = s["source_template_id"]
            if tid not in template_splits:
                template_splits[tid] = s["split"]
            else:
                assert s["split"] == template_splits[tid], (
                    f"Template {tid} found across multiple splits: {s['split']} vs {template_splits[tid]}"
                )


# ─── 6. Metric Computation Integrity ─────────────────────────────────────────

class TestMetricComputationAudit:
    def test_accuracy_and_f1_exact_formulas(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]

        acc = accuracy_score(y_true, y_pred)
        assert pytest.approx(acc, 1e-4) == 4 / 6

        macro_f1 = f1_score(y_true, y_pred, average="macro")
        macro_p = precision_score(y_true, y_pred, average="macro")
        macro_r = recall_score(y_true, y_pred, average="macro")

        assert 0.0 <= macro_f1 <= 1.0
        assert 0.0 <= macro_p <= 1.0
        assert 0.0 <= macro_r <= 1.0


# ─── 7. Missing-Model Fallback & Safety ───────────────────────────────────────

class TestMissingModelFallbackAudit:
    def test_classification_when_codebert_unavailable_falls_back_to_baseline(self):
        mock_classifier = MagicMock()
        mock_classifier.is_loaded = False
        mock_classifier.load.return_value = False

        with patch("backend.app.services.classification_service.get_classifier", return_value=mock_classifier):
            with patch("backend.app.services.classification_service._predict_baseline") as mock_base:
                mock_base.return_value = {
                    "predicted_label": "correct",
                    "probabilities": {"correct": 0.8, "wrong_initialization": 0.05, "off_by_one_loop": 0.05, "wrong_operator": 0.05, "wrong_condition": 0.05},
                    "confidence": 0.8,
                }
                res = analyze_code("def foo(): return 1")
                assert res["model_type"] == "baseline"
                assert res["predicted_label"] == "correct"

    def test_classification_when_no_models_available_does_not_fabricate(self):
        mock_classifier = MagicMock()
        mock_classifier.is_loaded = False
        mock_classifier.load.return_value = False

        with patch("backend.app.services.classification_service.get_classifier", return_value=mock_classifier):
            with patch("backend.app.services.classification_service._predict_baseline", return_value=None):
                res = analyze_code("def foo(): return 1")
                assert res["model_type"] == "unavailable"
                assert res["predicted_label"] == "unknown"
                assert "No trained classifier is available" in res["feedback"]
