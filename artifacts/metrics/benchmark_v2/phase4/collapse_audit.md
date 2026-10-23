# Phase 4 collapse audit
- Checksum: `6d2d6ab93918ef86...`
- Train labels: {'correct': 38, 'wrong_operator': 11, 'wrong_initialization': 10, 'wrong_condition': 10, 'off_by_one_loop': 5}
- Val labels: {'correct': 10, 'wrong_initialization': 2, 'wrong_operator': 2, 'wrong_condition': 2}
- Test labels (historical): {'correct': 18, 'wrong_operator': 5, 'off_by_one_loop': 4, 'wrong_initialization': 2, 'wrong_condition': 2}

## Observed (phase 3 checkpoint on test)
- Predicted distribution: {'correct': 31}
- Per-class recall: {'correct': 1.0, 'wrong_initialization': 0.0, 'off_by_one_loop': 0.0, 'wrong_operator': 0.0, 'wrong_condition': 0.0}
- Collapse warning: True