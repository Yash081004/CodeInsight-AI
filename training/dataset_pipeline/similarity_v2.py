"""Similarity pair generation for benchmark_v2 with explicit pair types."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def pair_key(code_a: str, code_b: str) -> Tuple[str, str]:
    ha, hb = code_hash(code_a), code_hash(code_b)
    return (ha, hb) if ha <= hb else (hb, ha)


def generate_similarity_pairs_v2(
    samples: List[dict],
    seed: int = 42,
    max_dissimilar_per_family_pair: int = 3,
    max_hard_negatives_per_family: int = 4,
) -> List[dict]:
    """
    Labels:
      1 = similar (same family, same behavior class — both correct or same-style)
      0 = dissimilar or hard negative (different behavior / different family)

    pair_type documents the relation (not semantic equivalence proof).
    """
    random.seed(seed)
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_family[s["family_id"]].append(s)

    pairs: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    pair_id = 0

    def add_pair(code_a, code_b, label, pair_type, fa, fb, split_tag):
        nonlocal pair_id
        pk = pair_key(code_a, code_b)
        if pk in seen:
            return
        seen.add(pk)
        pair_id += 1
        pairs.append(
            {
                "pair_id": f"P{pair_id:05d}",
                "code_a": code_a,
                "code_b": code_b,
                "label": label,
                "pair_type": pair_type,
                "family_a": fa,
                "family_b": fb,
                "split": split_tag,
                "source": f"{fa}|{fb}|{pair_type}",
            }
        )

    # Positives: two correct implementations in same family (different templates)
    for fam, group in by_family.items():
        correct = [g for g in group if g["error_label"] == "correct"]
        for i in range(len(correct)):
            for j in range(i + 1, len(correct)):
                sa, sb = correct[i], correct[j]
                if sa["source_template_id"] == sb["source_template_id"]:
                    continue
                sp_a = sa.get("split", "train")
                sp_b = sb.get("split", "train")
                split_tag = sp_a if sp_a == sp_b else "cross_split"
                add_pair(
                    sa["code"],
                    sb["code"],
                    1,
                    "equivalent_family_correct",
                    fam,
                    fam,
                    split_tag,
                )

    # Hard negatives: correct vs wrong in same family (high lexical overlap)
    for fam, group in by_family.items():
        correct = [g for g in group if g["error_label"] == "correct"]
        wrong = [g for g in group if g["error_label"] != "correct"]
        if not correct or not wrong:
            continue
        combos = [(c, w) for c in correct for w in wrong]
        random.shuffle(combos)
        for c, w in combos[:max_hard_negatives_per_family]:
            sp_a, sp_b = c.get("split"), w.get("split")
            split_tag = sp_a if sp_a == sp_b else "cross_split"
            add_pair(c["code"], w["code"], 0, "hard_negative_same_family", fam, fam, split_tag)

    # Dissimilar: different families (correct vs correct), capped
    families = sorted(by_family.keys())
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            fa, fb = families[i], families[j]
            ga = [x for x in by_family[fa] if x["error_label"] == "correct"]
            gb = [x for x in by_family[fb] if x["error_label"] == "correct"]
            if not ga or not gb:
                continue
            combos = [(a, b) for a in ga for b in gb]
            random.shuffle(combos)
            for a, b in combos[:max_dissimilar_per_family_pair]:
                sp_a, sp_b = a.get("split"), b.get("split")
                split_tag = sp_a if sp_a == sp_b else "cross_split"
                add_pair(a["code"], b["code"], 0, "cross_family_dissimilar", fa, fb, split_tag)

    return pairs
