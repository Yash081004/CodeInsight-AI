# Phase 13 — Review queue mismatch analysis

Generated: 2026-10-21T09:31:55.213812+00:00

## Root cause (evidence-based)

Phase 12 triage labeled **5** items `insufficient_evidence` because no CSV row matched `template_id + intended_label + mutation_type`. Reconciliation shows those queue entries refer to **generator mutations that still pass bundled tests** and were **never written** to `dataset.csv` (see `generate_benchmark.py`: `continue` after `review_queue.append`).

This is **not** missing CSV data; it is **queue-only excluded mutations**.

Four additional items match CSV rows but share **taxonomy collisions** (multiple mutations with the same label + mutation_type) or **stale queue reasons** vs hash-linked CSV samples.

## Summary by mismatch class

- **queue_only_excluded_mutation**: 5
- **stale_queue_reason_vs_csv_sample**: 1
- **taxonomy_collision_multiple_mutations**: 4

## Per-item trace

### `factorial_2` / `wrong_condition` / `base_case`
- **Class:** queue_only_excluded_mutation
- **CSV strict matches:** none
- Generator produced mutation(s) that still pass bundled tests; generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead.
- Matched 1 passing mutation variant(s); 0 CSV rows with same taxonomy key.
- Generator: pass=True — Changed base case from n<=1 to n<=0, causing extra recursion

### `fibonacci_1` / `wrong_operator` / `arithmetic_op`
- **Class:** queue_only_excluded_mutation
- **CSV strict matches:** none
- Generator produced mutation(s) that still pass bundled tests; generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead.
- Matched 1 passing mutation variant(s); 0 CSV rows with same taxonomy key.
- Generator: pass=True — Changed + to - in Fibonacci recurrence

### `fibonacci_1` / `wrong_condition` / `boundary_condition`
- **Class:** taxonomy_collision_multiple_mutations
- **CSV strict matches:** ['S0014']
- Multiple distinct mutations share the same error_label + mutation_type; CSV may contain one variant while queue flags another.
- CSV row hash-linked to generator: S0014 (tests_pass=False).
- Generator: pass=False — Changed n <= 0 to n < 0, fails for fib(0)

### `fibonacci_2` / `wrong_condition` / `boundary_condition`
- **Class:** taxonomy_collision_multiple_mutations
- **CSV strict matches:** ['S0016']
- Multiple distinct mutations share the same error_label + mutation_type; CSV may contain one variant while queue flags another.
- CSV row hash-linked to generator: S0016 (tests_pass=False).
- Generator: pass=True — Changed n <= 0 to n < 0, fails for fib(0)

### `is_prime_2` / `wrong_condition` / `boundary_condition`
- **Class:** taxonomy_collision_multiple_mutations
- **CSV strict matches:** ['S0024']
- Multiple distinct mutations share the same error_label + mutation_type; CSV may contain one variant while queue flags another.
- CSV row hash-linked to generator: S0024 (tests_pass=False).
- Generator: pass=True — Changed n < 2 to n < 1, incorrectly treating 1 as prime

### `palindrome_1` / `off_by_one_loop` / `loop_boundary`
- **Class:** queue_only_excluded_mutation
- **CSV strict matches:** none
- Generator produced mutation(s) that still pass bundled tests; generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead.
- Matched 1 passing mutation variant(s); 0 CSV rows with same taxonomy key.
- Generator: pass=True — Changed < to <= in pointer comparison, causes index issues

### `linear_search_2` / `wrong_initialization` / `index_init`
- **Class:** queue_only_excluded_mutation
- **CSV strict matches:** none
- Generator produced mutation(s) that still pass bundled tests; generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead.
- Matched 1 passing mutation variant(s); 0 CSV rows with same taxonomy key.
- Generator: pass=True — Changed starting index from 0 to 1

### `linear_search_2` / `off_by_one_loop` / `loop_boundary`
- **Class:** queue_only_excluded_mutation
- **CSV strict matches:** none
- Generator produced mutation(s) that still pass bundled tests; generate_benchmark.py skips adding them to dataset.csv and appends review_queue instead.
- Matched 1 passing mutation variant(s); 0 CSV rows with same taxonomy key.
- Generator: pass=True — Changed loop boundary to skip last element

### `bubble_sort_1` / `off_by_one_loop` / `range_endpoint`
- **Class:** taxonomy_collision_multiple_mutations
- **CSV strict matches:** ['S0059']
- Multiple distinct mutations share the same error_label + mutation_type; CSV may contain one variant while queue flags another.
- CSV row hash-linked to generator: S0059 (tests_pass=False).
- Generator: pass=False — Changed bubble sort inner loop boundary, causes index out of range

### `linear_search_1` / `off_by_one_loop` / `range_endpoint`
- **Class:** stale_queue_reason_vs_csv_sample
- **CSV strict matches:** ['S0047']
- Queue reason mutation_still_passes_tests but hash-linked CSV sample fails bundled tests today.
- Linked sample S0047 (train): Shortened range by 1, missing last element
- Generator: pass=False — Shortened range by 1, missing last element
