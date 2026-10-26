#!/usr/bin/env python3
"""End-to-end assembly on a header-only mock: no weights, no GPU, no docker.

Builds a mock of the quant from its real index / quantization_config.json (every tensor with its real
dtype and shape, zero bytes of data), fakes the GPU conversion the way tools/convert_natives.py would
rewrite each affected shard, mocks diffbot's pack (recipe/serve/serve.sh, a plan, Engram shards 47/48
from real headers) and the stock drafter shards (real headers), then runs prepare_pack.py with
--skip-convert --skip-download --no-docker and checks the result.

usage: test_offline.py --quant-meta <dir> --stock-drafter <dir> --engram <dir> [--diffbot-plan plan.json] [--keep DIR]
  --quant-meta   config.json, model.safetensors.index.json, quantization_config.json, tokenizer*.json of the quant
  --stock-drafter directory with (header-only is fine) model-00044..46-of-00048.safetensors of DeepSeek 1017
  --engram       directory with (header-only is fine) diffbot's model-00047/48-of-00048.safetensors
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import convert_natives as cn  # noqa: E402

DT = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.int16": "I16", "torch.int32": "I32",
      "torch.float32": "F32", "torch.float8_e4m3fn": "F8_E4M3", "torch.uint8": "U8"}
NBYTES = {"BF16": 2, "F16": 2, "I16": 2, "I32": 4, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "I8": 1}


def write_header_shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    off, h = 0, {}
    for k, (dt, shape) in tensors.items():
        n = NBYTES[dt]
        for d in shape:
            n *= d
        h[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    b = json.dumps(h).encode()
    path.write_bytes(struct.pack("<Q", len(b)) + b)


def build_quant_mock(meta: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in meta.iterdir():
        if f.suffix == ".json":
            shutil.copy(f, dst)
    wm = json.load(open(meta / "model.safetensors.index.json"))["weight_map"]
    ts = json.load(open(meta / "quantization_config.json"))["tensor_storage"]
    known = {k: t for v in ts.values() for k, t in v.get("stored_tensors", {}).items()}
    files: dict[str, dict] = collections.defaultdict(dict)
    for k, f in wm.items():
        t = known.get(k)
        if t:
            dt, shape = DT.get(t["dtype"], "BF16"), t["shape"]
        elif k.endswith("attn.wo_b.weight"):
            dt, shape = "BF16", [5120, 8192]
        elif k.endswith("shared_experts.w2.weight"):
            dt, shape = "BF16", [5120, 2304]
        elif ".engram.embed.weight" in k:
            dt, shape = "F8_E4M3", [1, 256]
        elif ".engram.embed.scale" in k:
            dt, shape = "F8_E8M0", [1, 8]
        elif ".engram.wkv.weight" in k:
            dt, shape = "F8_E4M3", [25600, 6144]
        elif ".engram.wkv.scale" in k:
            dt, shape = "F8_E8M0", [800, 192]
        else:
            dt, shape = "BF16", [1]
        files[f][k] = (dt, shape)
    # the real quant's index is stale: shards 2 and 15 physically hold layers.{1,14}.engram.{k,q}_weight
    # copies that the index attributes to the Engram shards; emulate that
    for layer, shard in ((1, wm.get("layers.1.engram.wkv.trellis")), (14, wm.get("layers.14.engram.wkv.trellis"))):
        if shard:
            files[shard][f"layers.{layer}.engram.k_weight"] = ("BF16", [4, 5120])
            files[shard][f"layers.{layer}.engram.q_weight"] = ("BF16", [4, 5120])
    for f, ten in files.items():
        write_header_shard(dst / f, ten)


def fake_convert(quant: Path, pack: Path, plan: dict) -> None:
    """What convert_natives.convert_shard would write, header-only."""
    pack.mkdir(parents=True, exist_ok=True)
    for shard, spec in plan.items():
        h = cn.shard_header(quant / shard)
        out = {k: (v["dtype"], v["shape"]) for k, v in h.items() if k not in set(spec["drop"])}
        for mod in spec["wo_a"]:
            out[mod + ".weight"] = ("F8_E4M3", [8192, 4096])
            out[mod + ".scale"] = ("F8_E8M0", [256, 128])
        for mod in spec["unquant"]:
            out[mod + ".weight"] = ("BF16", [128, 512] if mod.endswith(".indexer.wk") else [512, 5120])
        if spec.get("head"):
            out["head.weight"] = ("BF16", [129280, 5120])
        for name in spec["bf16"]:
            dt, shape = out.pop(name)
            out[name] = ("F8_E4M3", shape)
            out[name[: -len(".weight")] + ".scale"] = ("F8_E8M0", [shape[0] // 32, shape[1] // 32])
        write_header_shard(pack / shard, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-meta", required=True)
    ap.add_argument("--stock-drafter", required=True)
    ap.add_argument("--engram", required=True)
    ap.add_argument("--diffbot-plan", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--engram-dir-mode", action="store_true", help="pass the Engram shards via --engram instead of the diffbot folder")
    a = ap.parse_args()
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="dsv41-29-"))
    quant, diffbot, pack, stock = work / "quant", work / "diffbot", work / "pack", work / "stock"
    build_quant_mock(Path(a.quant_meta), quant)
    # diffbot mock
    (diffbot / "recipe" / "serve").mkdir(parents=True, exist_ok=True)
    (diffbot / "recipe" / "plans").mkdir(parents=True, exist_ok=True)
    (diffbot / "recipe" / "serve" / "serve.sh").write_text("#!/bin/bash\n")
    plan_src = Path(a.diffbot_plan) if a.diffbot_plan else None
    if plan_src and plan_src.is_file():
        shutil.copy(plan_src, diffbot / "recipe" / "plans" / "plan-1017-agent30-c512.json")
    else:
        fake = {f"language_model.model.layers.{L}.ffn.experts": list(range(100)) for L in range(43)}
        json.dump(fake, open(diffbot / "recipe" / "plans" / "plan-1017-agent30-c512.json", "w"))
    engram_dst = (work / "engram-elsewhere") if a.engram_dir_mode else diffbot
    engram_dst.mkdir(parents=True, exist_ok=True)
    for s in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        shutil.copy(Path(a.engram) / s, engram_dst / s)
    stock.mkdir(exist_ok=True)
    for n in (44, 45, 46):
        s = f"model-000{n}-of-00048.safetensors"
        shutil.copy(Path(a.stock_drafter) / s, stock / s)

    plan = cn.plan_shards(quant)
    n_woa = sum(len(d["wo_a"]) for d in plan.values())
    n_unq = sum(len(d["unquant"]) for d in plan.values())
    n_bf = sum(len(d["bf16"]) for d in plan.values())
    n_mtp = sum(1 for d in plan.values() for n in d["drop"] if n.startswith("mtp."))
    assert n_woa == 40, n_woa
    assert n_unq == 11, n_unq   # 4 indexer.wk + compressor wkv (layers 2, 8, 14, 20) + wgate (2, 8, 14; layer 20 has ratio 1, no gate)
    assert n_bf == 54, n_bf
    assert n_mtp > 0, "shard 43's mtp tensors must be dropped"
    assert "model-00042-of-00043.safetensors" not in plan, "drafter-only shard is not rewritten"
    assert sum(1 for d in plan.values() if d.get("head")) == 1
    print(f"plan: {len(plan)} shards rewritten; wo_a {n_woa}, unquant {n_unq}, bf16 {n_bf}, mtp dropped {n_mtp}, lm_head 1")
    fake_convert(quant, pack, plan)

    rc = subprocess.run([sys.executable, str(ROOT / "prepare_pack.py"), "--quant", str(quant), "--diffbot", str(diffbot),
                         "--pack", str(pack), "--stock-drafter", str(stock), "--skip-convert", "--skip-download",
                         "--no-docker", "--env", str(work / "none.env")]
                        + (["--engram", str(engram_dst)] if a.engram_dir_mode else [])).returncode
    assert rc == 0, f"prepare_pack.py exit {rc}"

    idx = json.load(open(pack / "model.safetensors.index.json"))["weight_map"]
    names = set(idx)
    assert not any("wo_a.slice." in n for n in names)
    assert not any(n.endswith((".indexer.wk.trellis", ".compressor.wkv.trellis", ".compressor.wgate.mul1")) for n in names)
    assert all(n.startswith("mtp.") for n, f in idx.items() if f.startswith("model-0004") and f.endswith("-of-00048.safetensors") and int(f[9:11]) < 47)
    assert "mtp.0.attn.wo_a.weight" in names and "mtp.0.ffn.experts.0.w1.weight" in names
    assert idx["layers.1.engram.wkv.weight"] == "model-00047-of-00048.safetensors"
    assert "layers.1.engram.wkv.trellis" not in names
    assert idx["layers.1.engram.k_weight"] == "model-00047-of-00048.safetensors", "stale physical copy dropped"
    assert idx["layers.10.attn.wo_a.weight"].endswith("-of-00043.safetensors") and "layers.10.attn.wo_a.scale" in names
    assert idx["layers.10.attn.wo_b.weight"] and "layers.10.attn.wo_b.scale" in names, "BF16 wo_b -> block-FP8"
    assert "layers.9.attn.wo_b.trellis" in names, "EXL3 wo_b stays EXL3"
    assert "layers.2.attn.indexer.wk.weight" in names
    assert "layers.14.attn.compressor.wkv.weight" in names and "layers.14.attn.compressor.wgate.weight" in names
    assert "head.trellis" not in names and idx["head.weight"].endswith("-of-00043.safetensors")
    dup = collections.Counter(idx.values())
    cfg = json.load(open(pack / "config.json"))["quantization_config"]
    assert cfg["mtp_experts"] == "source" and cfg["mtp_experts_start_layer"] == 40
    assert cfg["layer_bits"] == {str(L): 2 for L in range(18, 23)}, cfg["layer_bits"]
    L = cfg["non_routed_exl3"]["layers"]
    assert L["language_model.model.layers.0.attn.fused_wqa_wkv"]["bits"] == 6
    assert L["language_model.model.layers.5.attn.fused_wqa_wkv"]["bits"] == 5
    assert L["language_model.model.layers.11.ffn.shared_experts.gate_up_proj"]["bits"] == 4
    assert L["language_model.model.layers.0.ffn.shared_experts.down_proj"]["bits"] == 5
    assert "language_model.model.layers.10.ffn.shared_experts.down_proj" not in L, "converted w2 is not EXL3"
    assert "language_model.model.layers.10.attn.wo_b" not in L and L["language_model.model.layers.9.attn.wo_b"]["bits"] == 5
    assert not any(".compressor." in k for k in L), "vLLM builds the compressor linear with quant_config=None"
    assert L["language_model.model.layers.2.attn.indexer.wq_b"]["bits"] == 5
    assert "language_model.model.layers.2.attn.indexer.wk" not in L
    assert L["language_model.model.layers.1.engram.wkv"]["bits"] == 5 if "language_model.model.layers.1.engram.wkv" in L else True
    assert not any(int(k.split(".layers.")[1].split(".")[0]) >= 40 for k in L if ".layers." in k), "drafter layers are not EXL3"
    assert len(L) == 154, len(L)
    assert "language_model.lm_head" not in L
    assert cfg["non_routed_quantization"]["quant_method"] == "deepseek_v4_fp8"
    assert cfg["engram_embed_format"] == "int4_fp16s32"
    assert "modules" not in cfg["non_routed_exl3"]
    plan_out = json.load(open(pack / "plan.json"))
    assert all(len(plan_out[k]) == 0 for k in plan_out if int(k.split(".layers.")[1].split(".")[0]) in (18, 19, 20, 21, 22, 40, 41, 42))
    assert (diffbot / "recipe" / "plans" / "plan-2.9bpw-plan-1017-agent30-c512.json").is_file()
    info = json.load(open(pack / "PACK_INFO.json"))
    assert info["engram_dir"] == str(engram_dst), info["engram_dir"]
    print(f"ok: {info['shards']} shards, {info['tensors']} tensors, dense EXL3 modules {len(L)}, "
          f"K dist {dict(sorted(collections.Counter(v['bits'] for v in L.values()).items()))}")
    if not a.keep:
        shutil.rmtree(work)
    else:
        print("kept", work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
