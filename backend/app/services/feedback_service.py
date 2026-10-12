"""
CodeInsight AI - Feedback Service

Deterministic feedback templates mapped to classification labels.
No LLM API calls - all feedback is predefined.
"""

from backend.app.config import FEEDBACK_TEMPLATES, FRIENDLY_LABELS, ERROR_LABELS


def get_feedback(label: str) -> str:
    """Get the deterministic feedback template for a predicted label."""
    if label not in FEEDBACK_TEMPLATES:
        return (
            f"Unknown label: '{label}'. Expected one of: {', '.join(ERROR_LABELS)}. "
            "Please check the model configuration."
        )
    return FEEDBACK_TEMPLATES[label]


def get_friendly_label(label: str) -> str:
    """Get user-friendly display name for a label."""
    return FRIENDLY_LABELS.get(label, label)


def get_all_feedback() -> dict:
    """Return all feedback templates."""
    return dict(FEEDBACK_TEMPLATES)
