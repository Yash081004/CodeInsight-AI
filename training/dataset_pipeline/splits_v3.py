"""Family-grouped splits with label-coverage refinement (benchmark_v3)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from training.dataset_pipeline.splits import SPLITS, assign_groups, verify_group_isolation

ERROR_LABELS_ALL = (
    "correct",
    "wrong_initialization",
    "off_by_one_loop",
    "wrong_operator",
    "wrong_condition",
)


def labels_by_family(records: List[dict]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for r in records:
        out[r["family_id"]].add(r["error_label"])
    return dict(out)


def class_coverage_by_split(records: List[dict]) -> Dict[str, Dict[str, int]]:
    cov: Dict[str, Counter] = {sp: Counter() for sp in SPLITS}
    for r in records:
        if r.get("split") in SPLITS:
            cov[r["split"]][r["error_label"]] += 1
    return {sp: dict(cov[sp]) for sp in SPLITS}


def families_per_split(records: List[dict]) -> Dict[str, List[str]]:
    fs: Dict[str, set] = {sp: set() for sp in SPLITS}
    for r in records:
        if r.get("split") in SPLITS:
            fs[r["split"]].add(r["family_id"])
    return {sp: sorted(fs[sp]) for sp in SPLITS}


def refine_splits_for_label_coverage(
    mapping: Dict[str, str],
    family_labels: Dict[str, Set[str]],
    seed: int = 42,
    required_labels: Tuple[str, ...] = ERROR_LABELS_ALL,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Swap whole families between splits to improve per-split label presence.
    Does not split families across splits.
    """
    random.seed(seed)
    notes: List[str] = []
    mapping = dict(mapping)
    families = list(mapping.keys())

    def missing_labels() -> Dict[str, Set[str]]:
        present = {sp: set() for sp in SPLITS}
        for fam, sp in mapping.items():
            present[sp].update(family_labels.get(fam, set()))
        return {sp: set(required_labels) - present[sp] for sp in SPLITS}

    for _ in range(500):
        miss = missing_labels()
        if all(len(miss[sp]) == 0 for sp in SPLITS):
            break
        improved = False
        for target_split in SPLITS:
            if not miss[target_split]:
                continue
            need = miss[target_split]
            for label in list(need):
                donors = [
                    f
                    for f in families
                    if label in family_labels.get(f, set()) and mapping[f] != target_split
                ]
                random.shuffle(donors)
                for fam in donors:
                    src = mapping[fam]
                    # find swap candidate in target that does not remove unique label from target
                    targets = [f for f in families if mapping[f] == target_split and f != fam]
                    random.shuffle(targets)
                    for other in targets:
                        if label in family_labels.get(other, set()):
                            continue
                        mapping[fam], mapping[other] = mapping[other], mapping[fam]
                        notes.append(f"swap families {fam}<->{other} to bring {label} into {target_split}")
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break

    final_miss = missing_labels()
    for sp, labels in final_miss.items():
        if labels:
            notes.append(f"coverage_gap:{sp}:missing={sorted(labels)}")
    return mapping, notes


def assign_family_splits_v3(
    records: List[dict],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
) -> Tuple[Dict[str, str], List[str]]:
    families = sorted(set(r["family_id"] for r in records))
    fam_labels = labels_by_family(records)
    base = assign_groups(families, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    refined, notes = refine_splits_for_label_coverage(base, fam_labels, seed=seed + 1)
    return refined, notes


def assert_family_isolation(records: List[dict]) -> None:
    ok, errs = verify_group_isolation(records, "family_id")
    if not ok:
        raise ValueError("Family split isolation violated: " + "; ".join(errs[:5]))
