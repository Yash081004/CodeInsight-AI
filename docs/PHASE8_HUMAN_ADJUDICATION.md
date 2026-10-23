# Phase 8 — Human-adjudicated dataset & behavioral validation

## Status (honest)

| Item | State |
|------|--------|
| Review queue | 30 items in `data/benchmark_v3/review_decisions.json` (priority batch: 4 samples) |
| Human decisions | **None** — all `pending` |
| Reviewed eval slice | **Blocked** — `data/benchmark_reviewed/manifest.json` |
| Production model | Unchanged (`models/codebert_classifier/`) |
| benchmark_v3 CSV | Unchanged (checksum preserved) |

**Adjudication remains blocked until a human reviewer runs `decide`.**

---

## Priority batch (packets prepared, not decided)

| Sample | Review ID | Family | CSV label | Spec source |
|--------|-----------|--------|-----------|-------------|
| S0027 | V3RQ001 | palindrome | wrong_operator | `palindrome_1` |
| S0028 | V3RQ002 | palindrome | wrong_condition | `palindrome_1` |
| S0003 | V3RQ003 | factorial | off_by_one_loop | `factorial_1` |
| S0009 | V3RQ004 | factorial | off_by_one_loop | `factorial_3` |

Note: S0009 is **`off_by_one_loop`** in the CSV (not `wrong_condition`).

Artifacts:
- `data/benchmark_v3/review_packets/priority_batch.json`
- `data/benchmark_v3/review_fixtures/S*.json`

Regenerate packets:

```bash
py -3.13 -m training.phase8_prepare_review_batch
```

---

## Reviewer workflow (CLI)

```bash
# Full packet (no model predictions by default)
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027

# Optional model hints (clearly not ground truth)
py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027 --show-model-hints

# Record decision (example — do not run unless you are the reviewer)
py -3.13 -m training.label_review_cli --dataset benchmark_v3 decide \
  --id V3RQ001 --status accepted --rationale "Template tests fail; label matches mutation." --reviewer YOUR_NAME

# Verify infrastructure
py -3.13 -m training.label_review_cli --dataset benchmark_v3 verify
```

**Allowed statuses:** `pending`, `accepted`, `relabeled`, `rejected`, `needs_expert_review`  
**Rationale required:** `rejected`, `relabeled`, `needs_expert_review`  
**Reviewer:** `--reviewer NAME` or `--anonymous-review`

Decisions append `audit_history`; they do **not** edit `dataset.csv`.

---

## Behavioral validation (review-only)

Allowlisted samples only (`S0027`, `S0028`, `S0003`, `S0009`):

```bash
py -3.13 -m training.review_test_runner --sample-id S0003
py -3.13 -m training.review_test_runner --all-priority
```

- Subprocess, 2s timeout, fixed fixtures from spec-derived asserts.
- **Not** a sandbox for arbitrary user code; API `run_tests` remains disabled.
- Passing/failing listed tests does **not** automatically accept or reject labels.

---

## Export reviewed slice

```bash
py -3.13 -m training.build_reviewed_eval_slice
```

Exports only `accepted` / `relabeled` with rationale, reviewer, no train-family overlap, checksum in manifest. Fails closed with zero adjudicated samples.

---

## API (Evaluation Lab)

- `GET /api/evaluation/review-lab` — counts, items, export eligibility, blocked slice status
- `GET /api/evaluation/review-queue?limit=40` — same payload (dataset must be `benchmark_v3`)
- `GET /api/evaluation/review-packet?sample_id=S0027` — full packet (validated IDs only)

All queue items remain **pending** until a human runs `decide` (currently **30** in `review_decisions.json`).

---

## Next action for humans

1. Review priority packets (start with V3RQ001–V3RQ004).
2. Run behavioral tests; note insufficient coverage in rationale if needed.
3. Record decisions via CLI.
4. Re-run `build_reviewed_eval_slice` when at least one sample is accepted/relabeled.
