#!/usr/bin/env python3
"""Rewrite the quant's shards so every non-EXL3 module is in DeepSeek's own storage format.

diffbot's stack serves two formats: EXL3 trellis (the vllm-exl3 plugin) and DeepSeek's block-FP8 /
BF16 (vLLM's own DeepseekV41 FP8 path, declared as `non_routed_quantization`). Four modules of this
quant are neither loadable as EXL3 by that stack nor already in DeepSeek's format:

  module                          in the quant                  written as (DeepSeek 1017 layout)
  layers.N.attn.wo_a              8 EXL3 K=5 slices (o-groups)  wo_a.weight  F8_E4M3 [8192, 4096]
                                                                wo_a.scale   F8_E8M0 [256, 128]   (32x32 blocks)
  layers.N.attn.indexer.wk        EXL3 K=8 (4 layers)           wk.weight    BF16 [128, 512]
  layers.N.attn.wo_b (27 layers)  BF16 .weight (no scales)      wo_b.weight  F8_E4M3 [5120, 8192] + scale [160, 256]
  layers.N.ffn.shared_experts.w2  BF16 .weight (27 layers)      w2.weight    F8_E4M3 [5120, 2304] + scale [160, 72]
  head                            EXL3 K=6 (lm_head)            head.weight  BF16 [129280, 5120] (DeepSeek's layout)
  layers.{1,14}.engram.*          EXL3 wkv + BF16 q/k copies    dropped: diffbot's Engram shards carry the stock ones
  mtp.*                           K=4 EXL3 drafter              dropped: stock 1017 shards 44-46 carry the drafter

The EXL3 ones are reconstructed with ExLlamaV3 (exact for the quantized values); BF16 -> FP8 is a
quantization at the precision DeepSeek ships those tensors in. Shards that hold none of these are not
touched (prepare_pack.py hardlinks them). Output shards keep their file names; the index is rebuilt
afterwards (tools/vendor/regenerate_safetensors_index.py).

Runs inside diffbot's serving image (torch, safetensors, exllamav3 with its CUDA extension):
  python3 convert_natives.py --src <quant> --out <pack> [--device cuda:0] [--only-shards a,b]
  python3 convert_natives.py --src <quant> --plan          # header-only: which shards/tensors would change
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

BLOCK = 32
FP8_MAX = 448.0
WO_A_GROUPS = 8
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mul1", "mcg")

RE_WOA_SLICE = re.compile(r"^(layers\.\d+\.attn\.wo_a)\.slice\.(\d+)\.(trellis|suh|svh|mul1|mcg)$")
RE_WK = re.compile(r"^(layers\.\d+\.attn\.indexer\.wk)\.(trellis|suh|svh|mul1|mcg)$")
RE_BF16_NATIVE = re.compile(r"^(layers\.\d+\.(?:attn\.wo_b|ffn\.shared_experts\.w2))\.weight$")
# The Engram projection (layers 1 and 14): the served pack takes DeepSeek's FP8 wkv and the BF16
# q/k weights from diffbot's Engram shards 47/48 (stock 1017 tensors next to the int4 tables), so the
# quant's EXL3 wkv and its copies of q/k_weight are dropped to keep every tensor name unique.
RE_ENGRAM = re.compile(r"^layers\.\d+\.engram\.(wkv\.(trellis|suh|svh|mul1|mcg)|k_weight|q_weight)$")
# The quant's own DSpark drafter (K=4 EXL3) is replaced by the stock 1017 drafter shards; its tensors
# are dropped wherever they share a shard with main-stack tensors (drafter-only shards are not linked).
RE_MTP = re.compile(r"^mtp\.")
# lm_head: the quant packs it as EXL3 K=6 (`head.trellis` ...), but this vLLM's DeepSeek loader maps only
# the name `head.weight` onto lm_head, so it goes back to DeepSeek's BF16 [vocab, hidden] (exact for K=6).
RE_HEAD = re.compile(r"^head\.(trellis|suh|svh|mul1|mcg)$")


def shard_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def classify(h: dict) -> dict:
    """What one shard header needs: {"wo_a": [modules], "wk": [modules], "bf16": [names], "drop": [names]}.

    Decided from the shard's own header, never from the index: the quant's index is stale (it lists
    tensors that are not there and omits some that are), and vLLM loads what the shards hold.
    """
    spec = {"wo_a": set(), "wk": set(), "bf16": [], "drop": [], "head": False}
    for name, v in h.items():
        if RE_HEAD.match(name):
            spec["head"] = True
            spec["drop"].append(name)
            continue
        m = RE_WOA_SLICE.match(name)
        if m:
            spec["wo_a"].add(m.group(1))
            spec["drop"].append(name)
            continue
        m = RE_WK.match(name)
        if m:
            spec["wk"].add(m.group(1))
            spec["drop"].append(name)
            continue
        if RE_BF16_NATIVE.match(name) and v["dtype"] == "BF16":
            spec["bf16"].append(name)
            continue
        if RE_ENGRAM.match(name) or RE_MTP.match(name):
            spec["drop"].append(name)
    return {"wo_a": sorted(spec["wo_a"]), "wk": sorted(spec["wk"]), "bf16": sorted(spec["bf16"]),
            "drop": sorted(spec["drop"]), "head": spec["head"]}


def needs_work(h: dict) -> bool:
    s = classify(h)
    return bool(s["wo_a"] or s["wk"] or s["bf16"] or s["drop"])


def shard_role(h: dict) -> str:
    """'drafter' (only mtp.*: left out, the stock drafter replaces it), 'engram' (only layers.N.engram.*:
    left out, diffbot's shards replace it), else 'main'."""
    names = list(h)
    if names and all(RE_MTP.match(n) for n in names):
        return "drafter"
    if names and all(".engram." in n for n in names):
        return "engram"
    return "main"


def plan_shards(src: Path) -> dict[str, dict]:
    """{shard: classify(header)} for every main-stack shard file of the quant that needs rewriting."""
    out = {}
    for path in sorted(src.glob("*.safetensors")):
        h = shard_header(path)
        if shard_role(h) != "main":
            continue
        spec = classify(h)
        if spec["wo_a"] or spec["wk"] or spec["bf16"] or spec["drop"] or spec["head"]:
            out[path.name] = spec
    return out


def quant_shards(src: Path) -> dict[str, str]:
    """{shard file: role} for every *.safetensors in the quant directory."""
    return {p.name: shard_role(shard_header(p)) for p in sorted(src.glob("*.safetensors"))}


# ---------------------------------------------------------------------------
# DeepSeek block-FP8 (e4m3 values, ue8m0 power-of-two scale per 32x32 block)
# ---------------------------------------------------------------------------
def to_block_fp8(w):
    import torch

    out_f, in_f = w.shape
    assert out_f % BLOCK == 0 and in_f % BLOCK == 0, w.shape
    w = w.float()
    blocks = w.view(out_f // BLOCK, BLOCK, in_f // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))                                  # [out/32, in/32]
    # smallest power of two s with amax / s <= FP8_MAX; zero blocks get s = 1
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / FP8_MAX)), torch.zeros_like(amax))
    e = e.clamp(-127, 127)
    scale = torch.exp2(e)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(out_f, in_f)
    e8m0 = (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q.contiguous(), e8m0.contiguous()


def exl3_linear(tensors: dict, prefix: str, device):
    """LinearEXL3 over the packed tensors of one matrix; returns it and (in, out)."""
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3

    trellis = tensors[prefix + ".trellis"].to(device)
    suh = tensors[prefix + ".suh"].to(device).half()
    svh = tensors[prefix + ".svh"].to(device).half()
    kw = {}
    if prefix + ".mul1" in tensors:
        kw["mul1"] = tensors[prefix + ".mul1"].to(device).contiguous()
    elif prefix + ".mcg" in tensors:
        kw["mcg"] = tensors[prefix + ".mcg"].to(device).contiguous()
    lin = LinearEXL3(config=None, in_features=int(suh.numel()), out_features=int(svh.numel()),
                     trellis=trellis.contiguous(), suh=suh.contiguous(), svh=svh.contiguous(),
                     out_dtype=torch.half, **kw)
    return lin, int(suh.numel()), int(svh.numel())


def reconstruct(tensors: dict, prefix: str, device):
    """Dequantized matrix in checkpoint orientation [out, in] (fp32)."""
    lin, _, _ = exl3_linear(tensors, prefix, device)
    w = lin.get_weight_tensor()        # [in, out], original basis
    return w.t().float().contiguous()


def convert_shard(src: Path, out: Path, shard: str, spec: dict, device) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src / shard), "pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    drop = set(spec["drop"])
    added = {}
    stats = {"shard": shard, "wo_a": [], "wk": [], "bf16": [], "head": False, "err": {}}
    with torch.no_grad():
        for mod in spec["wo_a"]:
            parts = []
            for g in range(WO_A_GROUPS):
                parts.append(reconstruct(tensors, f"{mod}.slice.{g}", device))   # [1024, 4096] each
            w = torch.cat(parts, dim=0)                                           # [8192, 4096]: group g = rows g*1024..
            q, s = to_block_fp8(w)
            added[mod + ".weight"], added[mod + ".scale"] = q.cpu(), s.cpu()
            stats["wo_a"].append(mod)
            stats["err"][mod] = _rel_err(w, q, s)
            del parts, w
        for mod in spec["wk"]:
            w = reconstruct(tensors, mod, device)                                  # [128, 512]
            added[mod + ".weight"] = w.to(torch.bfloat16).cpu()
            stats["wk"].append(mod)
        if spec.get("head"):
            w = reconstruct(tensors, "head", device)                               # [129280, 5120]
            added["head.weight"] = w.to(torch.bfloat16).cpu()
            stats["head"] = True
            del w
        for name in spec["bf16"]:
            w = tensors[name].to(device).float()
            q, s = to_block_fp8(w)
            mod = name[: -len(".weight")]
            added[name], added[mod + ".scale"] = q.cpu(), s.cpu()
            stats["bf16"].append(name)
            stats["err"][mod] = _rel_err(w, q, s)
            del w
    keep = {k: v for k, v in tensors.items() if k not in drop and k not in added}
    keep.update(added)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (shard + ".tmp")
    save_file(keep, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, out / shard)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["tensors"] = len(keep)
    return stats


def _rel_err(w, q, s) -> float:
    import torch

    out_f, in_f = w.shape
    deq = q.float().view(out_f // BLOCK, BLOCK, in_f // BLOCK, BLOCK) * \
        s.view(torch.uint8).float().sub(127).exp2()[:, None, :, None]
    deq = deq.view(out_f, in_f)
    return float(((deq - w).norm() / w.norm().clamp_min(1e-12)).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only-shards", default="", help="comma list; default all affected")
    ap.add_argument("--plan", action="store_true", help="header-only listing, no GPU")
    ap.add_argument("--manifest", default="", help="write the plan/stats JSON here")
    a = ap.parse_args()
    src = Path(a.src)
    plan = plan_shards(src)
    if a.plan or not a.out:
        summary = {s: {k: (len(v) if isinstance(v, list) else v) for k, v in d.items()} for s, d in plan.items()}
        n_woa = sum(len(d["wo_a"]) for d in plan.values())
        n_wk = sum(len(d["wk"]) for d in plan.values())
        n_bf = sum(len(d["bf16"]) for d in plan.values())
        n_head = sum(1 for d in plan.values() if d.get("head"))
        print(f"{len(plan)} shards to rewrite: wo_a modules {n_woa}, indexer.wk {n_wk}, bf16 natives {n_bf}, lm_head {n_head}, "
              f"EXL3 tensors dropped {sum(len(d['drop']) for d in plan.values())}")
        if a.manifest:
            json.dump({"plan": plan}, open(a.manifest, "w"), indent=1)
        return 0
    import torch

    device = torch.device(a.device)
    only = {s for s in a.only_shards.split(",") if s}
    results = []
    for shard in sorted(plan):
        if only and shard not in only:
            continue
        print(f"{shard}: wo_a {len(plan[shard]['wo_a'])} wk {len(plan[shard]['wk'])} bf16 {len(plan[shard]['bf16'])}"
              f"{' head' if plan[shard].get('head') else ''} drop {len(plan[shard]['drop'])} ...",
              flush=True)
        st = convert_shard(src, Path(a.out), shard, plan[shard], device)
        worst = max(st["err"].values(), default=0.0)
        print(f"  -> {st['tensors']} tensors in {st['seconds']}s, worst FP8 block rel err {worst:.4f}", flush=True)
        results.append(st)
    if a.manifest:
        json.dump({"plan": plan, "results": results}, open(a.manifest, "w"), indent=1)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
