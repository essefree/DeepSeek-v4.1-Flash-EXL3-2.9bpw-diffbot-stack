#!/usr/bin/env python3
"""A copy of diffbot's recipe/serve/serve.sh that serves without an API key when API_KEY is unset.

Their script requires API_KEY and passes `--api-key` to vLLM, so every request needs a Bearer token.
This writes a copy with three exact-text edits (it refuses if their script changed under them):
  * the `API_KEY` requirement becomes optional,
  * `--api-key "$API_KEY"` is passed only when API_KEY is set,
  * REPO (their recipe directory) is pinned, since the copy lives outside their tree.
The warmup's `Authorization: Bearer` header stays: vLLM ignores it when no key is configured.

usage: patch_diffbot_serve.py <diffbot recipe dir> <output file>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EDITS = [
    (': "${API_KEY:?set API_KEY}"\n',
     'API_KEY=${API_KEY:-}   # [2.9bpw recipe] empty = no authentication\n'),
    ('--host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" \\\n',
     '--host 0.0.0.0 --port "$PORT" ${API_KEY:+--api-key "$API_KEY"} \\\n'),
]


def patch(text: str, recipe: Path) -> str:
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"diffbot's serve.sh changed: expected exactly one {old.strip()!r}")
        text = text.replace(old, new)
    old_repo = 'REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)\n'
    if text.count(old_repo) != 1:
        raise SystemExit("diffbot's serve.sh changed: REPO= line not found")
    text = text.replace(old_repo, f'REPO={str(recipe)!r}   # [2.9bpw recipe] pinned: this copy lives outside their tree\n')
    return text


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    recipe, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    src = recipe / "serve" / "serve.sh"
    if not src.is_file():
        raise SystemExit(f"{src} missing")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patch(src.read_text(), recipe))
    os.chmod(out, 0o755)
    print(f"{out}: diffbot's serve.sh with optional API_KEY (REPO={recipe})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
