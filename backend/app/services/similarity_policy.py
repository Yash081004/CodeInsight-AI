"""
Similarity decision policy (embedding score ≠ semantic equivalence).

Selected on validation pairs; see artifacts/metrics/similarity_hybrid_policy.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Tuned on data/similarity val split (grid over tf_low / tf_confirm); see training/similarity_policy_evaluation.py
DEFAULT_TF_LOW = 0.40
DEFAULT_TF_CONFIRM = 0.55


@dataclass(frozen=True)
class SimilarityDecision:
    is_similar: bool
    decision_model: str
    decision_policy: str
    review_recommended: bool
    reason: str


def hybrid_midband_decision(
    *,
    codebert_score: Optional[float],
    tfidf_score: Optional[float],
    threshold: float,
    tf_low: float = DEFAULT_TF_LOW,
    tf_confirm: float = DEFAULT_TF_CONFIRM,
) -> SimilarityDecision:
    """
    When CodeBERT exceeds threshold:
      - TF-IDF >= tf_confirm → accept (lexical/structural agreement).
      - TF-IDF <= tf_low → accept (rename-tolerant; CodeBERT-primary).
      - TF-IDF in (tf_low, tf_confirm) → reject (common false-positive band on unrelated algo pairs).
    If CodeBERT is below threshold → not similar.
    Without CodeBERT → fall back to TF-IDF vs threshold.
    """
    policy = "hybrid_midband_v1"

    if codebert_score is None:
        if tfidf_score is None:
            return SimilarityDecision(
                False,
                "none",
                policy,
                True,
                "No similarity scores available.",
            )
        ok = tfidf_score >= threshold
        return SimilarityDecision(
            ok,
            "tfidf",
            policy,
            False,
            "TF-IDF primary (CodeBERT unavailable).",
        )

    if codebert_score < threshold:
        return SimilarityDecision(
            False,
            "codebert",
            policy,
            False,
            f"CodeBERT score {codebert_score:.4f} below threshold {threshold:.2f}.",
        )

    if tfidf_score is None:
        return SimilarityDecision(
            True,
            "codebert",
            policy,
            True,
            "CodeBERT above threshold; TF-IDF unavailable — treat as unconfirmed embedding similarity.",
        )

    if tfidf_score >= tf_confirm:
        return SimilarityDecision(
            True,
            "hybrid_confirmed",
            policy,
            False,
            "CodeBERT and TF-IDF both support similarity (not proof of identical behavior).",
        )

    if tfidf_score <= tf_low:
        return SimilarityDecision(
            True,
            "hybrid_codebert_rename_tolerance",
            policy,
            True,
            "CodeBERT high with low TF-IDF (often renaming); embedding match only — verify behavior separately.",
        )

    return SimilarityDecision(
        False,
        "hybrid_rejected_mid_tfidf",
        policy,
        False,
        (
            f"CodeBERT {codebert_score:.4f} ≥ threshold but TF-IDF {tfidf_score:.4f} is in the "
            f"ambiguous band ({tf_low:.2f}, {tf_confirm:.2f}) — likely shared syntax, not equivalent logic."
        ),
    )


def decide_from_scores(
    *,
    codebert_score: Optional[float],
    tfidf_score: Optional[float],
    threshold: float,
    tf_low: float = DEFAULT_TF_LOW,
    tf_confirm: float = DEFAULT_TF_CONFIRM,
) -> Tuple[SimilarityDecision, Dict[str, Any]]:
    d = hybrid_midband_decision(
        codebert_score=codebert_score,
        tfidf_score=tfidf_score,
        threshold=threshold,
        tf_low=tf_low,
        tf_confirm=tf_confirm,
    )
    meta = {
        "decision_policy": d.decision_policy,
        "decision_model": d.decision_model,
        "review_recommended": d.review_recommended,
        "tf_low": tf_low,
        "tf_confirm": tf_confirm,
        "score_gap": (
            round(codebert_score - tfidf_score, 4)
            if codebert_score is not None and tfidf_score is not None
            else None
        ),
    }
    return d, meta
