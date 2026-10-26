#!/usr/bin/env python3
"""Turn the vendored pack tool's config.json into what diffbot's stack loads.

tools/vendor/qwen_pack_config.py writes the per-tensor K map the vllm-exl3 plugin reads, but keyed by
checkpoint tensor prefixes (`layers.3.attn.wq_a`). The plugin matches `non_routed_exl3.layers` keys
against vLLM *module* prefixes, which for DeepSeek-V4.1 in diffbot's vLLM are re-rooted and partly
fused:

  layers.N.attn.wq_a + layers.N.attn.wkv          -> language_model.model.layers.N.attn.fused_wqa_wkv
  layers.N.attn.compressor.wkv + .wgate           -> ...attn.compressor.fused_wkv_wgate
  layers.N.ffn.shared_experts.w1 + .w3            -> ...ffn.shared_experts.gate_up_proj
  layers.N.ffn.shared_experts.w2                  -> ...ffn.shared_experts.down_proj
  layers.N.<anything else>                        -> language_model.model.layers.N.<anything else>

Fused modules need one K; this refuses when the halves disagree. Entries at K=0 (tensors stored
natively) and anything outside the main stack (mtp.*, vision.*) are dropped: the FP8 delegate serves
those. Then the keys diffbot's pack carries are added: the FP8 delegate for non-EXL3 tensors
(`non_routed_quantization`), the source-format drafter (`mtp_experts: "source"` from layer 40), and
the int4 Engram tables.

usage: rewrite_config.py <pack_dir> [--base-bits 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

ROOT_PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head"
FUSED = {  # checkpoint module tail -> (vLLM module tail, shard index)
    "attn.wq_a": ("attn.fused_wqa_wkv", 0),
    "attn.wkv": ("attn.fused_wqa_wkv", 1),
    "attn.compressor.wkv": ("attn.compressor.fused_wkv_wgate", 0),
    "attn.compressor.wgate": ("attn.compressor.fused_wkv_wgate", 1),
    "ffn.shared_experts.w1": ("ffn.shared_experts.gate_up_proj", 0),
    "ffn.shared_experts.w3": ("ffn.shared_experts.gate_up_proj", 1),
    "ffn.shared_experts.w2": ("ffn.shared_experts.down_proj", 0),
}
NON_ROUTED_QUANTIZATION = {  # DeepSeek's own non-expert format, as diffbot's config.json declares it
    "quant_method": "deepseek_v4_fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "scale_fmt": "ue8m0",
    "weight_block_size": [32, 32],
    "expert_dtype": "fp4",
}
MTP_START_LAYER = 40


def map_key(key: str) -> str | None:
    """vLLM module prefix for a checkpoint module prefix; None = not served as EXL3."""
    if key == "head":
        return None  # reconstructed to BF16 head.weight by convert_natives.py (vLLM maps only that name)
    m = re.match(r"^layers\.(\d+)\.(.+)$", key)
    if not m:
        return None  # mtp.*, vision.*, ...
    layer, tail = m.group(1), m.group(2)
    if ".wo_a.slice." in tail or tail == "attn.indexer.wk" or tail.startswith("engram."):
        return None  # converted to DeepSeek's format by convert_natives.py / taken from the Engram shards
    tail = FUSED.get(tail, (tail, 0))[0]
    return f"{ROOT_PREFIX}layers.{layer}.{tail}"


def rewrite(q: dict, base_bits: int) -> dict:
    nr = q.get("non_routed_exl3") or {}
    layers_in = nr.get("layers") or {}
    merged: dict[str, set[int]] = collections.defaultdict(set)
    for key, spec in layers_in.items():
        bits = int(spec.get("bits", 0))
        if bits <= 0:
            continue
        new = map_key(key)
        if new is None:
            continue
        merged[new].add(bits)
    layers_out = {}
    for key, bits in sorted(merged.items()):
        if len(bits) != 1:
            raise SystemExit(f"fused module {key} has halves at different K {sorted(bits)}: cannot serve")
        k = bits.pop()
        if k not in (2, 3, 4, 5, 6):
            raise SystemExit(f"{key}: K={k} is outside what the plugin accepts for dense linears (2-6)")
        layers_out[key] = {"bits": k}
    layer_bits = {str(k): int(v) for k, v in (q.get("layer_bits") or {}).items()}
    out = {
        "quant_method": "exl3",
        "bits": int(q.get("bits", base_bits)),
        "codebook": str(q.get("codebook", "mul1")),
        "head_bits": int(q.get("head_bits", 6)),
        "layer_bits": layer_bits,
        "non_routed_exl3": {
            "codebook": str(nr.get("codebook", q.get("codebook", "mul1"))),
            "bits": 5,
            "layers": layers_out,
        },
        "non_routed_quantization": NON_ROUTED_QUANTIZATION,
        "weight_block_size": [32, 32],
        "mtp_experts": "source",
        "mtp_experts_start_layer": MTP_START_LAYER,
        "engram_embed_format": "int4_fp16s32",
        "source": ("Mia-layout EXL3 2.9 bpw quant of DeepSeek-V4.1-Flash-1017 served on diffbot's stack: "
                   "routed experts EXL3 (K=3 mul1, K=2 on layers 18-22), dense EXL3 per non_routed_exl3.layers, "
                   "wo_a / indexer.wk reconstructed and wo_b / shared w2 re-encoded in DeepSeek's block-FP8 "
                   "(tools/convert_natives.py), stock 1017 DSpark drafter (source format), diffbot int4 Engram"),
    }
    if q.get("native_quantization_config"):
        out["native_quantization_config"] = q["native_quantization_config"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path = Path(a.pack) / "config.json"
    cfg = json.loads(path.read_text())
    # the vendored tool may have written the block under text_config; the served config keeps it top-level
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if not q or "non_routed_exl3" not in q:
        raise SystemExit("run tools/vendor/qwen_pack_config.py first (no non_routed_exl3 in quantization_config)")
    new = rewrite(q, a.base_bits)
    cfg["quantization_config"] = new
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    dist = collections.Counter(v["bits"] for v in new["non_routed_exl3"]["layers"].values())
    print(f"dense EXL3 modules: {len(new['non_routed_exl3']['layers'])} (K distribution {dict(sorted(dist.items()))}); "
          f"expert layer_bits overrides: {new['layer_bits']}; drafter: source from layer {MTP_START_LAYER}")
    if a.dry_run:
        return 0
    backup = path.with_suffix(".json.pre-rewrite")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
