#!/usr/bin/env bash
# Build diffbot's xmoe and xmoe_do_i8 kernels with the K=2 mul1 instances this quant needs, using
# diffbot's own build scripts inside their serving image (no GPU needed, a few minutes).
#
# Output goes into diffbot's recipe tree, where their serve.sh (mounted at /opt/recipe) looks for
# kernels:  <DIFFBOT>/recipe/kernels/build/xmoe_k2/        -> serve.sh XMOE_EXT=xmoe_k2
#           <DIFFBOT>/recipe/kernels/build/k2/xmoe_do_i8/  -> EXL3_DO_MOD=xmoe_do_i8 EXL3_DO_DIR=/opt/recipe/kernels/build/k2/xmoe_do_i8
# diffbot's own builds (kernels/build/xmoe, kernels/build/xmoe_do_i8) are left untouched.
#
# usage: kernels/build.sh          (reads DIFFBOT and IMAGE from .env; IMAGE defaults to diffbot's tag)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
: "${DIFFBOT:?set DIFFBOT=<path to the diffbot pack> in .env}"
IMAGE=${IMAGE:-dsv41-flash-1017-exl3-3bpw-sm120}
K=$DIFFBOT/recipe/kernels
[ -d "$K/xmoe" ] || { echo "no $K/xmoe: DIFFBOT must point at a full download of diffbot's pack (recipe/ included)"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing: build it with $DIFFBOT/recipe/docker/build.sh"; exit 1; }

# A patched copy of the kernel tree: diffbot's sources stay pristine.
TREE=$ROOT/kernels/tree
rm -rf "$TREE"; mkdir -p "$TREE"
cp -r "$K/xmoe" "$K/xmoe_do_i8" "$K/build_xmoe.py" "$K/build_xmoe_do_i8.py" "$TREE/"
python3 "$HERE/add_k2_mul1.py" "$TREE"

run() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e TORCH_CUDA_ARCH_LIST=12.0a -e MAX_JOBS="${MAX_JOBS:-8}" "$@"; }
mkdir -p "$K/build/xmoe_k2" "$K/build/k2"
echo "== xmoe_k2 (group + multi-M + flat kernels, K=2/3 mcg + K=2/3 mul1)"
run -v "$TREE":/k -v "$K/build":/out -w /k -e XMOE_BUILD_DIR=/out --entrypoint python3 "$IMAGE" build_xmoe.py xmoe xmoe_k2
echo "== xmoe_do_i8 with (K=2, mul1) dispatch"
run -v "$TREE":/k -v "$K/build/k2":/out -w /k -e XMOE_BUILD_DIR=/out --entrypoint python3 "$IMAGE" build_xmoe_do_i8.py
ls -la "$K/build/xmoe_k2" "$K/build/k2/xmoe_do_i8" | grep -E "\.so|total"
echo "== done. serve.sh uses XMOE_EXT=xmoe_k2 EXL3_DO_DIR=/opt/recipe/kernels/build/k2/xmoe_do_i8"
