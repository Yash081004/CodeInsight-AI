# Phase 13 — Proposed review packet (no decisions recorded)

Grouped to minimize reviewer effort. **human_reviewed: false** for all items.

## Batch A — Queue-only mutations (not in CSV)

Reviewer needs: confirm whether mutation should enter a future benchmark with stronger tests, be relabeled, or remain excluded.

- **factorial_2** `wrong_condition` (base_case)
  - Passing mutation: Changed base case from n<=1 to n<=0, causing extra recursion
- **fibonacci_1** `wrong_operator` (arithmetic_op)
  - Passing mutation: Changed + to - in Fibonacci recurrence
- **palindrome_1** `off_by_one_loop` (loop_boundary)
  - Passing mutation: Changed < to <= in pointer comparison, causes index issues
- **linear_search_2** `wrong_initialization` (index_init)
  - Passing mutation: Changed starting index from 0 to 1
- **linear_search_2** `off_by_one_loop` (loop_boundary)
  - Passing mutation: Changed loop boundary to skip last element

## Batch B — CSV linked but queue reason or taxonomy ambiguous

- **fibonacci_1** `wrong_condition` — samples ['S0014'] — _taxonomy_collision_multiple_mutations_
- **fibonacci_2** `wrong_condition` — samples ['S0016'] — _taxonomy_collision_multiple_mutations_
- **is_prime_2** `wrong_condition` — samples ['S0024'] — _taxonomy_collision_multiple_mutations_
- **bubble_sort_1** `off_by_one_loop` — samples ['S0059'] — _taxonomy_collision_multiple_mutations_
- **linear_search_1** `off_by_one_loop` — samples ['S0047'] — _stale_queue_reason_vs_csv_sample_