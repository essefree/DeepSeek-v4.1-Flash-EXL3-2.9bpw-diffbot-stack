#!/usr/bin/env python3
"""Add K=2 mul1 (codebook 2) instances to diffbot's xmoe / xmoe_do_i8 kernel sources.

diffbot's pack is K=3 mul1 everywhere, so their `xmoe` build instantiates the group, multi-M and
flat MoE kernels for (K=2, mcg), (K=3, mcg) and (K=3, mul1) only, and `dof_gemm` (the decode-once
int8 prefill kernel) dispatches those three. This quant has K=2 mul1 experts on layers 18-22. The
kernels are templates on (K, N, codebook), with `dq_dispatch<bits, cb>` and `pack_s8x4<cb>` already
covering bits=2 and cb=2, so adding the instances is declarations, one comp unit and dispatch lines.

usage: add_k2_mul1.py <copy of diffbot's recipe/kernels>   (edits in place; idempotent)

The cold-path prefill kernel (`pb`, K=3 mul1 only) is not extended: the K=2 layers stay fully in
VRAM (plans/, tools/make_plan.py), so they never take the cold path.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "k2-mul1-instances"

INST_K2_CB2 = """#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"
#include "../exl3_moe_flat_kernel.cuh"

// K=2 mul1 (codebook 2): layers 18-22 of the 2.9 bpw quant. Mirrors exl3_moe_inst_k3_cb2.cu.
fp_exl3_moe_kernel exl3_moe_kernel_k2_n128_cb2() { return exl3_moe_kernel<2, 128, 2>; }
fp_exl3_moe_kernel exl3_moe_kernel_k2_n256_cb2() { return exl3_moe_kernel<2, 256, 2>; }
fp_exl3_moe_kernel exl3_moe_kernel_k2_n128_cb2_mt() { return exl3_moe_kernel<2, 128, 2, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k2_n256_cb2_mt() { return exl3_moe_kernel<2, 256, 2, true>; }
fp_exl3_moe_kernel exl3_moe_flat_kernel_k2_n128_cb2() { return exl3_moe_flat_kernel<2, 128, 2>; }
"""


def patch_text(path: Path, edits: list[tuple[str, str]], count: dict | None = None) -> None:
    text = path.read_text()
    if MARK in text:
        print(f"  {path.name}: already patched")
        return
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one occurrence of {old[:60]!r}, found {n}")
        text = text.replace(old, new)
    text = text.rstrip("\n") + f"\n// [{MARK}]\n"
    path.write_text(text)
    print(f"  {path.name}: patched ({len(edits)} edits)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    k = Path(sys.argv[1])
    xq = k / "xmoe" / "quant"
    for must in (xq / "exl3_moe.cu", xq / "comp_units" / "exl3_moe_instances.cuh",
                 xq / "comp_units" / "exl3_moe_inst_k3_cb2.cu", k / "xmoe_do_i8" / "dof_gemm.cu"):
        if not must.is_file():
            raise SystemExit(f"not diffbot's kernels tree: {must} missing")

    # 1. the comp unit
    inst = xq / "comp_units" / "exl3_moe_inst_k2_cb2.cu"
    if not inst.is_file():
        inst.write_text(INST_K2_CB2)
        print(f"  {inst.name}: written")
    else:
        print(f"  {inst.name}: exists")

    # 2. declarations of the getters the macro does not cover
    patch_text(xq / "comp_units" / "exl3_moe_instances.cuh", [(
        "fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb2_mt();\n",
        "fp_exl3_moe_kernel exl3_moe_kernel_k2_n128_cb2_mt();\n"
        "fp_exl3_moe_kernel exl3_moe_kernel_k2_n256_cb2_mt();\n"
        "fp_exl3_moe_kernel exl3_moe_flat_kernel_k2_n128_cb2();\n"
        "fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb2_mt();\n",
    )])

    # 3. dispatch tables ([K][cb-1][N_off]) and the flat-kernel gate
    patch_text(xq / "exl3_moe.cu", [
        ("    exl3_moe_kernel_k2_n128_cb1(), exl3_moe_kernel_k2_n256_cb1(), nullptr, nullptr,\n",
         "    exl3_moe_kernel_k2_n128_cb1(), exl3_moe_kernel_k2_n256_cb1(), exl3_moe_kernel_k2_n128_cb2(), exl3_moe_kernel_k2_n256_cb2(),\n"),
        ("    exl3_moe_kernel_k2_n128_cb1_mt(), exl3_moe_kernel_k2_n256_cb1_mt(), nullptr, nullptr,\n",
         "    exl3_moe_kernel_k2_n128_cb1_mt(), exl3_moe_kernel_k2_n256_cb1_mt(), exl3_moe_kernel_k2_n128_cb2_mt(), exl3_moe_kernel_k2_n256_cb2_mt(),\n"),
        ("(K == 2 || K == 3) && (cb_idx == 0 || K == 3) &&",
         "(K == 2 || K == 3) &&"),
        ("        fp_exl3_moe_kernel fk = K == 3 ? (cb_idx ? exl3_moe_flat_kernel_k3_n128_cb2() : exl3_moe_flat_kernel_k3_n128_cb1())\n"
         "                                       : exl3_moe_flat_kernel_k2_n128_cb1();\n",
         "        fp_exl3_moe_kernel fk = K == 3 ? (cb_idx ? exl3_moe_flat_kernel_k3_n128_cb2() : exl3_moe_flat_kernel_k3_n128_cb1())\n"
         "                                       : (cb_idx ? exl3_moe_flat_kernel_k2_n128_cb2() : exl3_moe_flat_kernel_k2_n128_cb1());\n"),
    ])

    # 4. decode-once prefill kernel dispatch
    patch_text(k / "xmoe_do_i8" / "dof_gemm.cu", [(
        "    else if (bits == 2 && cb == 1) launch_v<MODE, 2, 1>(a, 0, dev, st);\n",
        "    else if (bits == 2 && cb == 1) launch_v<MODE, 2, 1>(a, 0, dev, st);\n"
        "    else if (bits == 2 && cb == 2) launch_v<MODE, 2, 2>(a, 0, dev, st);\n",
    )])
    print("K=2 mul1 instances added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
