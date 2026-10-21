# Phase 14 — Specification approval (pending human)

All five priority families remain **`draft`** / **`human_reviewed: false`**.

| Family | Checklist command |
|--------|-------------------|
| palindrome | `py -3.13 -m training.spec_approval_cli checklist --family palindrome` |
| fibonacci | `py -3.13 -m training.spec_approval_cli checklist --family fibonacci` |
| linear_search | `py -3.13 -m training.spec_approval_cli checklist --family linear_search` |
| factorial | `py -3.13 -m training.spec_approval_cli checklist --family factorial` |
| gcd | `py -3.13 -m training.spec_approval_cli checklist --family gcd` |

Approval (human only):

`py -3.13 -m training.spec_approval_cli approve --family <id> --reviewer <NAME> --notes "..."`

Recorded ambiguities appear in checklist JSON — resolve or explicitly accept before approve.
