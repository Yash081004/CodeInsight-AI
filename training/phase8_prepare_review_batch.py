"""Materialize priority review packets (no decisions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.review_workflow import PRIORITY_SAMPLE_IDS, build_review_packet, load_decisions


def main() -> int:
    data = load_decisions("benchmark_v3")
    by_sample = {d["sample_id"]: d for d in data.get("decisions", []) if d.get("sample_id")}
    packets = []
    for sid in sorted(PRIORITY_SAMPLE_IDS, key=lambda x: ["S0027", "S0028", "S0003", "S0009"].index(x)):
        d = by_sample.get(sid)
        if not d:
            print(f"[WARN] no review row for {sid}")
            continue
        packets.append(build_review_packet(d, include_model_hints=False))
    out = PROJECT_ROOT / "data" / "benchmark_v3" / "review_packets" / "priority_batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_name": "phase8_priority",
        "human_adjudicated": False,
        "all_status_pending": True,
        "note": "Model predictions omitted by default; use build_review_packet(include_model_hints=True) for optional section.",
        "packets": packets,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} ({len(packets)} packets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
