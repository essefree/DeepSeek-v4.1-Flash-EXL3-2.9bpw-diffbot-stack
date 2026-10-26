#!/usr/bin/env python3
"""A copy of diffbot's recipe/serve/serve.sh that serves without an API key when API_KEY is unset.

Their script requires API_KEY and passes `--api-key` to vLLM, so every request needs a Bearer token.
This writes a copy with three exact-text edits (it refuses if their script changed under them):
  * the `API_KEY` requirement becomes optional,
  * `--api-key "$API_KEY"` is passed only when API_KEY is set,
  * REPO (their recipe directory) is pinned, since the copy lives outside their tree,
  * the sitecustomize mount points at this recipe's overlay/sitecustomize.py, which runs diffbot's
    and then adds the hooks this quant needs (EXL3 shared experts).
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
# (pattern, replacement, what, required): each must match at most once, wherever it sits on its line, so
# a revision of their script that moves things around still patches; a required one that is missing refuses.
EDITS = [
    (re.compile(r'^: "\$\{API_KEY:\?[^}]*\}"[ \t]*\n', re.M),
     'API_KEY=${API_KEY:-}   # [2.9bpw recipe] empty = no authentication\n',
     'the `: "${API_KEY:?...}"` requirement', False),
    (re.compile(r'--api-key "\$API_KEY"'),
     '${API_KEY:+--api-key "$API_KEY"}',
     'the `--api-key "$API_KEY"` argument', False),   # absent in a hand-edited copy: then nothing to do
    (re.compile(r'--served-model-name[ \t]+(?:"\$SERVED"|\$SERVED|[^ \t\\]+)'),
     '--served-model-name "$SERVED"',
     'the `--served-model-name` argument', True),       # a hand-edited copy may hardcode a name; SERVED must win
    (re.compile(r'^REPO=\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\." && pwd\)[ \t]*\n', re.M),
     None,  # filled with the pinned recipe path
     'the REPO= line', True),
    (re.compile(r'-v "\$REPO/patches/vllm/sitecustomize\.py":/usr/lib/python3\.12/sitecustomize\.py:ro'),
     "SITECUSTOMIZE",  # filled with this recipe's overlay/sitecustomize.py (it loads diffbot's first)
     'the sitecustomize mount', True),
]


def patch(text: str, recipe: Path, overlay: Path | None = None) -> tuple[str, list[str]]:
    notes = []
    for pattern, new, what, required in EDITS:
        hits = pattern.findall(text)
        if len(hits) > 1 or (required and not hits):
            raise SystemExit(f"diffbot's serve.sh differs from what this patch expects: {what} found {len(hits)} times "
                             f"(run: grep -n 'API_KEY\\|served-model-name\\|^REPO=' {recipe}/serve/serve.sh)")
        if not hits:
            notes.append(f"{what}: not present (hand-edited copy?), nothing to change")
            continue
        if new is None:
            new = f'REPO={str(recipe)!r}   # [2.9bpw recipe] pinned: this copy lives outside their tree\n'
        elif new == "SITECUSTOMIZE":
            site = (overlay or Path(__file__).resolve().parents[1] / "overlay") / "sitecustomize.py"
            new = f'-v "{site}":/usr/lib/python3.12/sitecustomize.py:ro'
        text = pattern.sub(lambda m: new, text, count=1)
    if 'API_KEY=${API_KEY:-}' not in text and '${API_KEY:?' not in text:
        text = text.replace("set -u\n", "set -u\nAPI_KEY=${API_KEY:-}   # [2.9bpw recipe]\n", 1)
    return text, notes


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    recipe, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    src = recipe / "serve" / "serve.sh"
    if not src.is_file():
        raise SystemExit(f"{src} missing")
    out.parent.mkdir(parents=True, exist_ok=True)
    text, notes = patch(src.read_text(), recipe)
    out.write_text(text)
    os.chmod(out, 0o755)
    for n in notes:
        print(f"  note: {n}")
    print(f"{out}: diffbot's serve.sh with optional API_KEY, SERVED honoured, our sitecustomize (REPO={recipe})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
