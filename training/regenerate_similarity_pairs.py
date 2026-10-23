"""
Regenerate similarity pairs from an existing classification dataset.csv.

Use when pairs.csv split tags are stale but classification data is valid.

Usage:
    py -3.13 -m training.regenerate_similarity_pairs
"""

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.generate_dataset import SEED, _generate_similarity_pairs


def main() -> int:
    clf_path = PROJECT_ROOT / "data" / "classification" / "dataset.csv"
    sim_dir = PROJECT_ROOT / "data" / "similarity"
    pairs_path = sim_dir / "pairs.csv"

    if not clf_path.exists():
        print(f"ERROR: {clf_path} not found")
        return 1

    with open(clf_path, newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    backup_dir = PROJECT_ROOT / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if pairs_path.exists():
        backup = backup_dir / f"pairs_{ts}.csv"
        shutil.copy2(pairs_path, backup)
        print(f"Backed up pairs to {backup}")

    _generate_similarity_pairs(samples, sim_dir, seed=SEED)
    print("[OK] pairs.csv regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
