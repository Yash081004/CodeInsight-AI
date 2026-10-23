"""

Production routing between CodeBERT, baseline, and transparent spec assists.



Collapse is detected by softmax margin (top1 − top2), not by which class won.

"""



from __future__ import annotations



from typing import Any, Dict, Optional, Tuple



# Production CodeBERT checkpoint exhibits ~flat softmax (margin ~0.02–0.04).

COLLAPSED_MARGIN = 0.05

BASELINE_PREFER_MARGIN = 0.10





def probability_margin(probabilities: Dict[str, float]) -> float:

    if not probabilities:

        return 0.0

    vals = sorted(probabilities.values(), reverse=True)

    if len(vals) < 2:

        return float(vals[0]) if vals else 0.0

    return float(vals[0] - vals[1])





def is_collapsed(probabilities: Dict[str, float], threshold: float = COLLAPSED_MARGIN) -> bool:

    return probability_margin(probabilities) < threshold





def _review_status_for_ml(

    *,

    primary_source: str,

    cb_collapsed: bool,

    has_problem_statement: bool,

    spec_status: Optional[str],

) -> str:

    if primary_source == "spec_rule_assist":

        return "spec_assisted"

    if primary_source == "baseline_fallback":

        return "baseline_routed"

    if cb_collapsed:

        if has_problem_statement and spec_status in (

            "vague_spec",

            "no_rule_applicable",

            "consistent",

            None,

        ):

            return "inconclusive"

        return "review_required"

    return "model_primary"





def route_classification(

    *,

    codebert: Optional[Dict[str, Any]],

    baseline: Optional[Dict[str, Any]],

    spec_hint: Optional[Dict[str, Any]],

    spec_assessment: Optional[Dict[str, Any]] = None,

    has_problem_statement: bool,

) -> Tuple[Dict[str, Any], str, str]:

    """

    Returns (result_dict, model_type, model_info).

    result_dict has predicted_label, probabilities, confidence, routing.

    """

    cb_margin = probability_margin(codebert["probabilities"]) if codebert else 0.0

    bl_margin = probability_margin(baseline["probabilities"]) if baseline else 0.0

    cb_collapsed = codebert is not None and is_collapsed(codebert["probabilities"])

    spec_status = (spec_assessment or {}).get("status")



    routing_notes: list[str] = []



    high_conf_violation = (

        spec_hint

        and spec_hint.get("confidence") == "high"

        and spec_status == "violation"

    )



    # Spec assist only when ML collapsed, user supplied a statement, and a high-confidence violation matched

    if high_conf_violation and has_problem_statement and cb_collapsed:

        ctx = codebert or baseline

        routing_notes.append(

            f"CodeBERT margin {cb_margin:.3f} below collapse threshold ({COLLAPSED_MARGIN}); "

            f"applied spec rule {spec_hint.get('rule_id')} (deterministic, not model output)."

        )

        review_status = _review_status_for_ml(

            primary_source="spec_rule_assist",

            cb_collapsed=cb_collapsed,

            has_problem_statement=has_problem_statement,

            spec_status=spec_status,

        )

        return (

            {

                "predicted_label": spec_hint["label"],

                "probabilities": ctx["probabilities"] if ctx else {},

                "confidence": float(ctx["confidence"]) if ctx else 0.0,

                "routing": {

                    "primary_source": "spec_rule_assist",

                    "codebert_margin": cb_margin,

                    "baseline_margin": bl_margin,

                    "codebert_predicted_label": codebert.get("predicted_label") if codebert else None,

                    "baseline_predicted_label": baseline.get("predicted_label") if baseline else None,

                    "spec_hint": spec_hint,

                    "spec_assessment_status": spec_status,

                    "score_applies_to_displayed_label": False,

                    "review_status": review_status,

                    "notes": routing_notes,

                },

            },

            "spec_rule_assist",

            "Deterministic spec check (problem statement + code); softmax scores are CodeBERT context only.",

        )



    # Prefer baseline when CodeBERT collapsed and baseline is more separated

    if codebert and baseline and cb_collapsed and bl_margin >= BASELINE_PREFER_MARGIN:

        routing_notes.append(

            f"CodeBERT collapsed (margin {cb_margin:.3f}); baseline margin {bl_margin:.3f}."

        )

        review_status = _review_status_for_ml(

            primary_source="baseline_fallback",

            cb_collapsed=cb_collapsed,

            has_problem_statement=has_problem_statement,

            spec_status=spec_status,

        )

        return (

            {

                "predicted_label": baseline["predicted_label"],

                "probabilities": baseline["probabilities"],

                "confidence": baseline["confidence"],

                "routing": {

                    "primary_source": "baseline_fallback",

                    "codebert_margin": cb_margin,

                    "baseline_margin": bl_margin,

                    "codebert_would_predict": codebert["predicted_label"],

                    "spec_assessment_status": spec_status,

                    "score_applies_to_displayed_label": True,

                    "review_status": review_status,

                    "notes": routing_notes,

                },

            },

            "baseline_fallback",

            "TF-IDF baseline (CodeBERT output was low-discrimination on this input).",

        )



    if codebert:

        routing_notes.append(f"CodeBERT primary (margin {cb_margin:.3f}, collapsed={cb_collapsed}).")

        if has_problem_statement and spec_status == "consistent":

            routing_notes.append("Deterministic spec check: implementation consistent with statement.")

        elif has_problem_statement and spec_status in ("vague_spec", "no_rule_applicable"):

            routing_notes.append(

                f"Problem statement present but spec assessment is {spec_status}; ML label only."

            )

        review_status = _review_status_for_ml(

            primary_source="codebert_finetuned",

            cb_collapsed=cb_collapsed,

            has_problem_statement=has_problem_statement,

            spec_status=spec_status,

        )

        return (

            {

                "predicted_label": codebert["predicted_label"],

                "probabilities": codebert["probabilities"],

                "confidence": codebert["confidence"],

                "routing": {

                    "primary_source": "codebert_finetuned",

                    "codebert_margin": cb_margin,

                    "baseline_margin": bl_margin,

                    "collapsed": cb_collapsed,

                    "spec_assessment_status": spec_status,

                    "score_applies_to_displayed_label": True,

                    "review_status": review_status,

                    "notes": routing_notes,

                },

            },

            "codebert_finetuned",

            "Fine-tuned CodeBERT",

        )



    if baseline:

        return (

            {

                "predicted_label": baseline["predicted_label"],

                "probabilities": baseline["probabilities"],

                "confidence": baseline["confidence"],

                "routing": {

                    "primary_source": "baseline",

                    "baseline_margin": bl_margin,

                    "score_applies_to_displayed_label": True,

                    "review_status": "baseline_routed",

                },

            },

            "baseline",

            "TF-IDF + Logistic Regression Baseline",

        )



    raise ValueError("No classifier result")


