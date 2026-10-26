#!/usr/bin/env python3
"""Where the quant is, the same way the first recipe takes it: MODEL=<dir | hf-org/name>.

  a local directory   MODEL=/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw   (served from there)
  a Hugging Face repo MODEL=essefree/my-exl3-quant   -> downloaded into <recipe>/models/<org>__<name>
  QUANT=<dir> (the earlier name of the setting) is accepted as a local directory.

The served pack is assembled next to the quant as <quant>-served unless PACK is set (same filesystem,
so the unchanged shards are hardlinks, not copies).

usage: resolve_model.py [--env .env] [--model M] [--pack P] [--shell | --json]
  --shell prints `MODEL_SOURCE=... QUANT=... PACK=... HF_MODEL_REPO=...` for `eval` in a shell script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def absolute(p: str) -> Path:
    p = os.path.expanduser(p)
    return Path(p) if os.path.isabs(p) else (ROOT / p).resolve()


def resolve(model: str | None, env: dict[str, str], pack: str | None = None) -> dict:
    model = (model or env.get("MODEL") or env.get("QUANT") or "").strip()
    if not model:
        raise ValueError("MODEL not set: ./serve.sh --model <dir | hf-org/name>, or MODEL= in .env")
    if model.startswith(("/", "./", "../", "~")) or Path(model).is_dir():
        source, quant, repo = "local", absolute(model), ""
    elif HF_RE.match(model):
        source, repo = "hf", model
        quant = ROOT / "models" / model.replace("/", "__")
    else:
        raise ValueError(f"MODEL={model!r} is neither an existing directory nor a Hugging Face repo id (org/name)")
    pack = pack or env.get("PACK") or ""
    pack_dir = absolute(pack) if pack else Path(str(quant).rstrip("/") + "-served")
    return {"MODEL": model, "MODEL_SOURCE": source, "QUANT": str(quant), "PACK": str(pack_dir), "HF_MODEL_REPO": repo}


def quant_complete(quant: Path) -> tuple[bool, str]:
    idx = quant / "model.safetensors.index.json"
    if not (quant / "config.json").is_file() or not idx.is_file():
        return False, "config.json / model.safetensors.index.json missing"
    wm = json.loads(idx.read_text())["weight_map"]
    missing = sorted({f for f in wm.values() if not (quant / f).is_file()})
    if missing:
        return False, f"{len(missing)} shards named by the index are missing (e.g. {missing[0]})"
    return True, f"{len(set(wm.values()))} shards"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--model")
    ap.add_argument("--pack")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        r = resolve(a.model, read_env(Path(a.env)), a.pack)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.shell:
        for k, v in r.items():
            print(f"{k}='{v}'")
    else:
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
