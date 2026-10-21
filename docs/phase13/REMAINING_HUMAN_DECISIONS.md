# Phase 13 — Remaining human decisions

No automatic label edits or `human_reviewed` flags were applied in this phase.

1. **Review queue (10 items)** — Adjudicate via `py -3.13 -m training.label_review_cli`:
   - **Batch A (5):** queue-only mutations that pass bundled tests and were never exported to CSV.
   - **Batch B (5):** taxonomy collisions or stale `mutation_still_passes_tests` vs hash-linked CSV samples.

2. **Specification packets (5 families)** — Edit and approve drafts in `data/specifications/packets/*.authoring_packet.json`, then promote catalog entries to `review_status: human_reviewed` with reviewer id.

3. **Benchmark v4 cut** — After specs + queue policy: re-run `py -3.13 -m training.benchmark_v4_build` and confirm non-zero eligible rows before any training experiment.

4. **Bundled tests** — Decide whether finite generator tests are sufficient per family or need extension before accepting ambiguous mutations into v4.
