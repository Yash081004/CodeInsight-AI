# Specification catalog schema (v1)

- JSON Schema: [`schemas/problem_specification.v1.json`](../../schemas/problem_specification.v1.json)
- Catalog file: [`data/specifications/catalog.v1.json`](../../data/specifications/catalog.v1.json)
- API: `training/spec_catalog.py`

## Review status

| Status | Meaning |
|--------|---------|
| `missing` | Legacy CSV row; no catalog entry |
| `draft` | Registry excerpt; **not** human-reviewed |
| `machine_suggested` | Automated proposal awaiting review |
| `human_reviewed` | Reviewer recorded in provenance |

Legacy datasets load unchanged; `expected_behavior` in CSV remains test-outcome flags only (`passes_all_tests` / `fails_tests`).

## Compatibility

- No changes to `dataset.csv` columns or production routing.
- `legacy_row_has_spec()` returns true only for `human_reviewed` catalog entries.
