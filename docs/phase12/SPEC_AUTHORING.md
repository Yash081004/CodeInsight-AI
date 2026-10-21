# Problem specification authoring template (v1)

Use for **new** or **human-reviewed** entries in `data/specifications/catalog.v1.json`.

```yaml
spec_id: spec.<family_id>.v1
family_id: <family_id>
source_template_ids: [<template_ids>]
problem_statement: |
  Describe intended behavior in plain language (inputs, output, preconditions).
input_assumptions:
  - e.g. n is a non-negative integer
expected_output_summary: one line
boundary_cases:
  - empty input
  - single element
provenance:
  source: human_reviewer
  author: <name or id>
  review_status: human_reviewed  # only after real review
  created_at: ISO-8601
associated_sample_ids: []  # optional links
```

**Do not** derive `problem_statement` from `problem_type` alone or from faulty code.

## Priority families to specify first (coverage impact)
1. **palindrome** — queue + ambiguous fault taxonomy  
2. **fibonacci** — multiple queue mutations  
3. **linear_search** — index/off-by-one queue items  
4. **factorial** — baseline arithmetic family  
5. **gcd** — common beginner template (no queue flag but high utility)
