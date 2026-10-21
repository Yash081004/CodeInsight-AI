# Phase 11 — Review queue packet (not human-reviewed)

These items are **verified** `mutation_still_passes_tests` flags. No label changes without `training.label_review_cli decide`.

| Priority | template_id | intended_label | mechanical check | human needed |
|----------|-------------|----------------|------------------|--------------|
| medium | factorial_2 | wrong_condition | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| high | fibonacci_1 | wrong_operator | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| high | fibonacci_1 | wrong_condition | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| medium | fibonacci_2 | wrong_condition | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| medium | is_prime_2 | wrong_condition | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| high | palindrome_1 | off_by_one_loop | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| high | linear_search_2 | wrong_initialization | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| high | linear_search_2 | off_by_one_loop | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| medium | bubble_sort_1 | off_by_one_loop | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |
| medium | linear_search_1 | off_by_one_loop | Bundled generator tests still pass | Confirm fault taxonomy vs intended_label |

CLI: `py -3.13 -m training.label_review_cli --dataset benchmark_v3 list`
Packet: `py -3.13 -m training.label_review_cli --dataset benchmark_v3 packet --sample S0027`