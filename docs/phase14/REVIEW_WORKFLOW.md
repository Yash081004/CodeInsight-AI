# Phase 14 — Review workflow commands

## Sample-based review (existing CSV rows)

```powershell
py -3.13 -m training.label_review_cli --dataset benchmark_v3 list
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --id V3RQ001
py -3.13 -m training.label_review_cli --dataset benchmark_v3 verify
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --reviewer alice
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide --id V3RQ001 --status rejected --rationale "..." --anonymous-review
```

Decisions append `audit_history` in `data/benchmark_v3/review_decisions.json`. **dataset.csv is not modified.**

## Queue mutation review (Batch A/B — no CSV sample_id)

Stable IDs: `artifacts/metrics/phase14/mutation_registry.json` (`mutation_id` field).

```powershell
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue list
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue packet --id QMRQ001
py -3.13 -m training.label_review_cli --dataset benchmark_v3 queue decide --id QMRQ001 --status needs_expert_review --rationale "..." --reviewer alice
```

Decisions: `data/benchmark_v3/queue_mutation_decisions.json`.

### Batch mapping

| Batch | QMRQ ids (see `queue list`) | Phase 13 class |
|-------|----------------------------|----------------|
| A | Queue-only excluded mutations | `queue_only_excluded_mutation` |
| B | CSV-linked / taxonomy / stale reason | collision or stale |

## Specification approval (draft → human_reviewed)

```powershell
py -3.13 -m training.spec_approval_cli list
py -3.13 -m training.spec_approval_cli checklist --family palindrome
py -3.13 -m training.spec_approval_cli approve --family palindrome --reviewer NAME --notes "..."
```

**Do not run `approve` until a real reviewer resolves checklist ambiguities.**

## Evidence semantics

| Observation | Does NOT imply |
|-------------|----------------|
| Bundled tests fail | Label is correct |
| Bundled tests pass | Code is logically correct |
| Approved family spec | Every sample label in family is correct |
