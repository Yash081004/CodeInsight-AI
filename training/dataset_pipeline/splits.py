"""Group-based split assignment and leakage checks."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


SPLITS = ("train", "val", "test")


def assign_groups(
    group_ids: List[str],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
) -> Dict[str, str]:
    """
    Assign each unique group_id to train/val/test.
    All records sharing a group_id must use the same split.
    """
    random.seed(seed)
    unique = sorted(set(group_ids))
    shuffled = list(unique)
    random.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return {}

    train_end = max(1, int(train_ratio * n)) if n >= 3 else max(1, n - 2)
    val_end = train_end + (max(1, int(val_ratio * n)) if n >= 3 else 1)
    if val_end >= n:
        val_end = n - 1 if n > 1 else n
    if train_end >= n:
        train_end = n - 2 if n > 2 else 1

    mapping: Dict[str, str] = {}
    for i, gid in enumerate(shuffled):
        if i < train_end:
            mapping[gid] = "train"
        elif i < val_end:
            mapping[gid] = "val"
        else:
            mapping[gid] = "test"
    return mapping


def verify_group_isolation(records: List[dict], group_key: str = "family_id") -> Tuple[bool, List[str]]:
    """Ensure no group_id appears in more than one split."""
    group_split: Dict[str, str] = {}
    errors: List[str] = []
    for r in records:
        gid = r.get(group_key, "")
        sp = r.get("split", "")
        if gid in group_split and group_split[gid] != sp:
            errors.append(f"Group {gid} spans {group_split[gid]} and {sp}")
        else:
            group_split[gid] = sp
    return len(errors) == 0, errors


def detect_pair_leakage(
    pairs: List[dict],
    sample_split_by_code_hash: Dict[str, str],
    code_hash_fn,
) -> Tuple[bool, List[str]]:
    """Pairs tagged train/val/test must not join codes from different splits."""
    errors: List[str] = []
    for p in pairs:
        split = p.get("split", "")
        if split not in SPLITS:
            continue
        ha = code_hash_fn(p.get("code_a", ""))
        hb = code_hash_fn(p.get("code_b", ""))
        sa = sample_split_by_code_hash.get(ha)
        sb = sample_split_by_code_hash.get(hb)
        if sa and sb and sa != sb:
            errors.append(f"Pair {p.get('pair_id')}: endpoints in {sa} vs {sb} tagged {split}")
        if sa and sa != split:
            errors.append(f"Pair {p.get('pair_id')}: tagged {split} but code_a in {sa}")
        if sb and sb != split:
            errors.append(f"Pair {p.get('pair_id')}: tagged {split} but code_b in {sb}")
    return len(errors) == 0, errors


def family_split_counts(records: List[dict], group_key: str = "family_id") -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        out[r.get("split", "?")][r.get(group_key, "?")] += 1
    return {k: dict(v) for k, v in out.items()}
