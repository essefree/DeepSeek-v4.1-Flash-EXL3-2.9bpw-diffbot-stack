#!/usr/bin/env python3
"""A copy of diffbot's recipe/serve/serve.sh that serves without an API key when API_KEY is unset.

Their script requires API_KEY and passes `--api-key` to vLLM, so every request needs a Bearer token.
This writes a copy with three exact-text edits (it refuses if their script changed under them):
  * the `API_KEY` requirement becomes optional,
  * `--api-key "$API_KEY"` is passed only when API_KEY is set,
  * REPO (their recipe directory) is pinned, since the copy lives outside their tree.
Matched by pattern, so a revision of their script that reflows those lines still patches.
The warmup's `Authorization: Bearer` header stays: vLLM ignores it when no key is configured.

usage: patch_diffbot_serve.py <diffbot recipe dir> <output file>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import re

# (pattern, replacement, what): each must match exactly once, wherever it sits on its line, so a
# revision of their script that moves things around still patches, and one that changed them refuses.
EDITS = [
    (re.compile(r'^: "\$\{API_KEY:\?[^}]*\}"[ \t]*\n', re.M),
     'API_KEY=${API_KEY:-}   # [2.9bpw recipe] empty = no authentication\n',
     'the `: "${API_KEY:?...}"` requirement'),
    (re.compile(r'--api-key "\$API_KEY"'),
     '${API_KEY:+--api-key "$API_KEY"}',
     'the `--api-key "$API_KEY"` argument'),
    (re.compile(r'^REPO=\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\." && pwd\)[ \t]*\n', re.M),
     None,  # filled with the pinned recipe path
     'the REPO= line'),
]


def patch(text: str, recipe: Path) -> str:
    for pattern, new, what in EDITS:
        hits = pattern.findall(text)
        if len(hits) != 1:
            raise SystemExit(f"diffbot's serve.sh differs from what this patch expects: {what} found {len(hits)} times "
                             f"(run: grep -n 'API_KEY\\|^REPO=' {recipe}/serve/serve.sh)")
        if new is None:
            new = f'REPO={str(recipe)!r}   # [2.9bpw recipe] pinned: this copy lives outside their tree\n'
        text = pattern.sub(lambda m: new, text, count=1)
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
