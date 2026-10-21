# Phase 12 — Go / no-go for next model training

**Decision: NO-GO** for CodeBERT training experiments until:

1. At least **5 priority families** have `human_reviewed` specifications in `catalog.v1.json`.
2. Review queue items with `mechanically_verifiable` + no NL spec are adjudicated via `label_review_cli`.
3. A **new versioned** benchmark split (v4+) is cut with frozen test and zero family leakage.

Rationale: Phase 11 showed collapse with only `[TASK] problem_type`; v3 has no NL specs in CSV. Template tests pass on many fault rows — labels are not reliably behavior-distinguished.

Triage summary: {
  "mechanically_verifiable": 0,
  "requires_reviewer_intended_behavior": 5,
  "requires_expert_decision": 0,
  "insufficient_evidence": 5
}

This phase did **not** train models or modify production.
