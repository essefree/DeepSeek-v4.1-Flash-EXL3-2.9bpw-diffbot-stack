#!/usr/bin/env python3
"""Assemble the served pack: the 2.9 bpw quant in the form diffbot's stack loads.

  PACK/
    model-000NN-of-00043.safetensors   the quant's shards: hardlinked, except the ones that held wo_a
                                       slices / indexer.wk / BF16 wo_b, w2 / the Engram projection,
                                       which tools/convert_natives.py rewrote (DeepSeek block-FP8, BF16)
    model-00044..46-of-00048.safetensors  stock DeepSeek-V4.1-Flash-1017 drafter (hardlinked)
    model-00047..48-of-00048.safetensors  diffbot's int4 Engram tables (hardlinked)
    config.json                        plugin vocabulary (tools/vendor + tools/rewrite_config.py)
    model.safetensors.index.json       rebuilt from the shard headers
    plan.json                          host-expert plan (also installed into diffbot's recipe/plans/)
    PACK_INFO.json                     what went in

Steps that need a GPU (the conversion) run in diffbot's image through docker. Everything else is
header-only Python on the host. Re-runnable: finished steps are skipped.

usage: prepare_pack.py [--env .env] [--skip-download] [--skip-convert] [--no-docker] [--force-links]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
VENDOR = TOOLS / "vendor"
sys.path.insert(0, str(TOOLS))
import convert_natives  # noqa: E402  (plan_shards, header-only)

STOCK_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
STOCK_REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DRAFTER_SHARDS = tuple(f"model-000{n}-of-00048.safetensors" for n in (44, 45, 46))
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def link(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not force:
            try:
                if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
                    return "exists"
            except OSError:
                pass
            if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
                return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        os.symlink(os.path.realpath(src), dst)
        return "symlink"


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def engram_format(path: Path) -> str:
    h = header(path)
    for k, v in h.items():
        if k.endswith(".engram.embed.weight"):
            return "int4" if v["dtype"] == "U8" and v["shape"][1] == 128 else "fp8"
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(HERE / ".env"))
    ap.add_argument("--quant"), ap.add_argument("--diffbot"), ap.add_argument("--pack")
    ap.add_argument("--stock-drafter"), ap.add_argument("--image"), ap.add_argument("--diffbot-plan")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-convert", action="store_true", help="converted shards are already in PACK")
    ap.add_argument("--no-docker", action="store_true", help="do not check the image (tests)")
    ap.add_argument("--force-links", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    env = read_env(Path(a.env))
    get = lambda key, flag: Path(flag) if flag else (Path(env[key]) if env.get(key) else None)
    quant, diffbot, pack = get("QUANT", a.quant), get("DIFFBOT", a.diffbot), get("PACK", a.pack)
    stock = get("STOCK_DRAFTER", a.stock_drafter) or (HERE / "drafter-1017")
    image = a.image or env.get("IMAGE") or "dsv41-flash-1017-exl3-3bpw-sm120"
    plan_name = a.diffbot_plan or env.get("DIFFBOT_PLAN") or "plan-1017-agent30-c512.json"
    for name, p in (("QUANT", quant), ("DIFFBOT", diffbot), ("PACK", pack)):
        if p is None:
            raise SystemExit(f"{name} not set (in {a.env} or --{name.lower()})")

    # ---- 1. inputs
    q_index = quant / "model.safetensors.index.json"
    if not (quant / "config.json").is_file() or not q_index.is_file():
        raise SystemExit(f"{quant}: config.json / model.safetensors.index.json missing")
    rec = diffbot / "recipe"
    if not (rec / "serve" / "serve.sh").is_file() or not (rec / "plans" / plan_name).is_file():
        raise SystemExit(f"{diffbot}: not a full download of diffbot's pack (recipe/serve/serve.sh, recipe/plans/{plan_name})")
    for s in ENGRAM_SHARDS:
        if not (diffbot / s).is_file():
            raise SystemExit(f"{diffbot / s} missing (diffbot's int4 Engram shard)")
        if engram_format(diffbot / s) != "int4":
            raise SystemExit(f"{diffbot / s} does not hold int4 Engram tables")
    if not a.no_docker and not a.skip_convert:
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
            raise SystemExit(f"docker image {image} missing: build it with {rec}/docker/build.sh")
    pack.mkdir(parents=True, exist_ok=True)
    wm = json.loads(q_index.read_text())["weight_map"]
    quant_shards = sorted(set(wm.values()))
    mtp_only = {s for s in quant_shards if all(k.startswith("mtp.") for k, f in wm.items() if f == s)}
    engram_named = {s for s in quant_shards if s in ENGRAM_SHARDS}
    log(f"quant: {len(quant_shards)} shards in the index, {len(mtp_only)} drafter-only (dropped), "
        f"{len(engram_named)} Engram (replaced by diffbot's)")

    # ---- 2. stock drafter
    stock.mkdir(parents=True, exist_ok=True)
    missing = [s for s in DRAFTER_SHARDS if not (stock / s).is_file()]
    if missing:
        if a.skip_download:
            raise SystemExit(f"stock drafter shards missing under {stock}: {missing}")
        hf = shutil.which("hf") or shutil.which("huggingface-cli")
        if not hf:
            raise SystemExit("need the Hugging Face CLI: pip install -U 'huggingface_hub[hf_transfer]'")
        cmd = [hf, "download", STOCK_REPO, "--revision", STOCK_REV, "--local-dir", stock]
        for s in missing:
            cmd += ["--include", s]
        run(cmd, env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")})
    for s in DRAFTER_SHARDS:
        h = header(stock / s)
        if not all(k.startswith("mtp.") for k in h):
            raise SystemExit(f"{stock / s}: not a DeepSeek drafter shard (non-mtp tensors)")

    # ---- 3. conversion plan + GPU conversion
    plan = convert_natives.plan_shards(quant)
    affected = sorted(plan)
    log(f"conversion: {len(affected)} shards hold wo_a slices / indexer.wk / BF16 natives / Engram projection")
    manifest = pack / "convert_manifest.json"
    done = {s for s in affected if (pack / s).is_file() and not (pack / s).is_symlink()
            and not _same_file(pack / s, quant / s)}
    todo = [s for s in affected if s not in done]
    if todo and a.skip_convert:
        raise SystemExit(f"--skip-convert but {len(todo)} converted shards are missing in {pack}: {todo[:3]}...")
    if todo:
        log(f"converting {len(todo)} shards in {image} on {a.device} (~17 GB of output) ...")
        run(["docker", "run", "--rm", "--gpus", "all", "--ipc", "host",
             "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
             "-v", f"{quant}:/quant:ro", "-v", f"{pack}:/pack", "-v", f"{TOOLS}:/tools:ro",
             "--entrypoint", "python3", image, "/tools/convert_natives.py", "--src", "/quant", "--out", "/pack",
             "--device", a.device, "--only-shards", ",".join(todo), "--manifest", "/pack/convert_manifest.json"])
    for s in affected:
        if not (pack / s).is_file():
            raise SystemExit(f"converted shard missing: {pack / s}")

    # ---- 4. links
    linked = 0
    for s in quant_shards:
        if s in plan or s in mtp_only or s in engram_named:
            continue
        link(quant / s, pack / s, a.force_links)
        linked += 1
    for s in DRAFTER_SHARDS:
        link(stock / s, pack / s, a.force_links)
    for s in ENGRAM_SHARDS:
        link(diffbot / s, pack / s, a.force_links)
    for f in COPY_FILES:
        if (quant / f).is_file():
            shutil.copy2(quant / f, pack / f)
    log(f"linked {linked} quant shards unchanged, "
        f"{len(DRAFTER_SHARDS)} drafter, {len(ENGRAM_SHARDS)} Engram; copied {', '.join(f for f in COPY_FILES if (quant / f).is_file())}")

    # ---- 5. config (plugin vocabulary) and index
    py = sys.executable
    run([py, "-I", VENDOR / "qwen_pack_scan.py", pack, "--out", pack / "pack_scan.json"], stdout=subprocess.DEVNULL)
    run([py, "-I", VENDOR / "qwen_pack_config.py", pack, "--scan", pack / "pack_scan.json"])
    run([py, TOOLS / "rewrite_config.py", pack])
    run([py, "-I", VENDOR / "regenerate_safetensors_index.py", pack])

    # ---- 6. plan
    run([py, TOOLS / "make_plan.py", rec / "plans" / plan_name, pack / "plan.json"])
    installed = rec / "plans" / f"plan-2.9bpw-{plan_name}"
    shutil.copy2(pack / "plan.json", installed)
    log(f"plan installed as {installed} (serve.sh PLAN={installed.name})")

    # ---- 7. verify
    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    names = set(idx)
    problems = []
    for bad in ("wo_a.slice.", ".indexer.wk.trellis", ".engram.wkv.trellis"):
        hits = [n for n in names if bad in n]
        if hits:
            problems.append(f"{len(hits)} unconverted tensors like {hits[0]}")
    if not any(n.startswith("mtp.") for n in names):
        problems.append("no mtp.* tensors (drafter missing)")
    if not any(n.endswith(".engram.embed.weight") for n in names):
        problems.append("no Engram tables")
    cfg = json.loads((pack / "config.json").read_text())["quantization_config"]
    for key in ("layer_bits", "non_routed_exl3", "non_routed_quantization", "mtp_experts", "engram_embed_format"):
        if key not in cfg:
            problems.append(f"config.json lacks quantization_config.{key}")
    total = sum(os.path.getsize(pack / s) for s in set(idx.values()))
    info = {
        "quant": str(quant), "diffbot": str(diffbot), "stock_drafter": {"repo": STOCK_REPO, "revision": STOCK_REV},
        "converted_shards": affected, "dropped_quant_shards": sorted(mtp_only | engram_named),
        "plan": {"from": plan_name, "installed": str(installed)},
        "tensors": len(idx), "shards": len(set(idx.values())), "bytes": total,
        "convert_manifest": str(manifest) if manifest.is_file() else None,
        "assembled": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (pack / "PACK_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    log(f"pack: {info['shards']} shards, {info['tensors']} tensors, {total / 2**30:.1f} GiB referenced")
    if problems:
        for p in problems:
            log("PROBLEM " + p)
        return 1
    log(f"OK -> {pack}")
    return 0


def _same_file(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
        return sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
