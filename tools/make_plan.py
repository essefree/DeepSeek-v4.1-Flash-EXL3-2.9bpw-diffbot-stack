#!/usr/bin/env python3
"""Host-expert plan for this quant from one of diffbot's plans.

diffbot's plans (`recipe/plans/*.json`) list, per MoE layer, the expert ids that stay in pinned host RAM.
They were built from the 1017 router's routing counts, which this quant shares (its router gates are the
stock ones), so the ids carry over. Two edits:

  * the K=2 layers keep every expert in VRAM: diffbot's cold-path prefill kernel and CPU runtime exist for
    K=3 mul1 only, and the K=2 layers are the smallest (1.6 GiB/rank each);
  * the drafter layers (40-42) are the stock 1017 drafter in source format, not EXL3: no entries.

usage: make_plan.py <diffbot plan.json> <out.json> [--vram-layers 18-22] [--extra-cold N]
  --extra-cold N: also demote the N experts that are next in line... not possible from a plan file (it has
  no ranking), so this script never adds experts; use diffbot's EXL3_EXPERT_STATS + make_plan.py for that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

K2_LAYERS = "18-22"
DRAFTER_LAYERS = (40, 41, 42)


def parse_layers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def layer_of(key: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", key)
    if not m:
        raise ValueError(f"not a layer key: {key}")
    return int(m.group(1))


def derive(plan: dict, vram_layers: set[int]) -> tuple[dict, dict]:
    out: dict = {}
    removed: dict[int, int] = {}
    for key, ids in plan.items():
        layer = layer_of(key)
        if layer in vram_layers or layer in DRAFTER_LAYERS:
            if ids:
                removed[layer] = len(ids)
            out[key] = []
        else:
            out[key] = sorted(int(i) for i in ids)
    return out, removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--vram-layers", default=K2_LAYERS, help="layers kept entirely in VRAM")
    a = ap.parse_args()
    plan = json.load(open(a.src))
    vram = parse_layers(a.vram_layers)
    out, removed = derive(plan, vram)
    json.dump(out, open(a.out, "w"))
    before = sum(len(v) for v in plan.values())
    after = sum(len(v) for v in out.values())
    print(f"{a.out}: {after} host experts (from {before}; {before - after} moved back to VRAM in layers "
          f"{sorted(removed)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
