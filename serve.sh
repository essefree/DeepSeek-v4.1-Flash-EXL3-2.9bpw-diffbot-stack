#!/usr/bin/env bash
# Serve the assembled pack through diffbot's recipe/serve/serve.sh with the settings this quant needs.
#
# Everything else (image, patches, mounts, API, CPU_EXPERTS, 1M context, DSpark k=5, warmup) is theirs;
# their script documents every knob and all of them can still be set in the environment.
#
#   ./serve.sh --model /data/models/my-exl3-quant     # a directory
#   ./serve.sh --model essefree/my-exl3-quant          # a Hugging Face repo (fetched)
#   ./serve.sh                                         # MODEL= from .env
# No authentication unless API_KEY is set (API_KEY=secret ./serve.sh -> clients send it as the Bearer
# token); diffbot's script requires a key, so a patched copy of it runs (tools/patch_diffbot_serve.py).
# Like the first recipe's start.sh, this does the whole path on the first run: fetch the quant
# (HF), assemble the pack (prepare_pack.py: stock drafter download + GPU conversion, ~30 min once),
# build the K=2 kernels (kernels/build.sh), then serve. NO_AUTO_PREPARE=1 only checks and stops.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_ARG="$2"; shift 2 ;;
        --model=*) MODEL_ARG="${1#--model=}"; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument $1 (only --model <dir | hf-org/name>)"; exit 2 ;;
    esac
done
[ -f "$ROOT/.env" ] || { echo "no .env (cp env.example .env)"; exit 1; }
set -a; source "$ROOT/.env"; set +a
export API_KEY=${API_KEY:-}
: "${DIFFBOT:?set DIFFBOT in .env (the downloaded diffbot pack)}"
model_args=()
[ -n "$MODEL_ARG" ] && model_args=(--model "$MODEL_ARG")
resolved="$(python3 "$ROOT/tools/resolve_model.py" --env "$ROOT/.env" ${model_args[@]+"${model_args[@]}"} --shell)" || exit 2
eval "$resolved"
export PACK
REC=$DIFFBOT/recipe
PLAN_NAME=${PLAN_NAME:-plan-2.9bpw-${DIFFBOT_PLAN:-plan-1017-agent30-c512.json}}
need_kernels=0
[ -d "$REC/kernels/build/xmoe_k2" ] && [ -d "$REC/kernels/build/k2/xmoe_do_i8" ] || need_kernels=1
if [ "${NO_AUTO_PREPARE:-0}" = "1" ]; then
    [ -f "$PACK/PACK_INFO.json" ] && [ -f "$REC/plans/$PLAN_NAME" ] || { echo "pack $PACK not assembled: ./prepare_pack.py --model '$MODEL'"; exit 1; }
    [ "$need_kernels" = 0 ] || { echo "K=2 kernels missing: bash kernels/build.sh"; exit 1; }
else
    # Always: the pack must reflect the current recipe (a converter fix re-converts only the shards it
    # concerns; the rest is header scans and a config / index / plan rebuild, seconds on a finished pack).
    echo "[serve] prepare_pack.py --model '$MODEL' (re-converts only what changed)"
    python3 "$ROOT/prepare_pack.py" --env "$ROOT/.env" --model "$MODEL" --pack "$PACK"
    if [ "$need_kernels" = 1 ]; then
        echo "[serve] K=2 mul1 kernels not built yet -> kernels/build.sh"
        bash "$ROOT/kernels/build.sh"
    fi
fi

# The pack's parent is what their serve.sh mounts (MODELS_DIR); the drafter is inside the pack.
export PACK MODELS_DIR=${MODELS_DIR:-$(dirname "$PACK")}
export IMAGE=${IMAGE:-dsv41-flash-1017-exl3-3bpw-sm120}
export SERVED=${SERVED:-deepseek-v4.1-flash-2.9bpw} PORT=${PORT:-8000} NAME=${NAME:-dsv41-exl3-29}
export PLAN=$PLAN_NAME HOST_CACHE=${HOST_CACHE:-512}
# K=2 mul1 instances for layers 18-22 (kernels/build.sh)
export XMOE_EXT=xmoe_k2 EXL3_DO=1 DO_INT8=1 EXL3_DO_MOD=xmoe_do_i8 EXL3_DO_DIR=/opt/recipe/kernels/build/k2/xmoe_do_i8
# The drafter is the stock 1017 head inside the pack (source format), not a separate folder.
export DS41_DRAFTER=released
# diffbot's FP8 fast paths assume FP8 dense tensors; this quant's attention/shared/head are EXL3.
# They check the dtype and skip, but keep them off so the log is unambiguous. wo_a, wo_b and the
# shared down-proj ARE block-FP8 here (convert_natives.py), so DS41_WOA_FP8 / DENSE_GEMV could apply
# to those; measure before turning them on.
export DS41_DENSE_GEMV=${DS41_DENSE_GEMV:-0} DS41_DENSE_ONE_COPY=${DS41_DENSE_ONE_COPY:-0}
export DS41_WOA_FP8=${DS41_WOA_FP8:-0} DS41_FP8_LMHEAD=${DS41_FP8_LMHEAD:-0}
export LINEAR_BACKEND=${LINEAR_BACKEND-}          # no Marlin MXFP8 dense repack (nothing to repack)
export DENSE_HYBRID_M=${DENSE_HYBRID_M:-0}
# diffbot's serve.sh with API_KEY optional (regenerated when their script changes)
DSERVE=$ROOT/.cache/diffbot-serve.sh
if [ ! -f "$DSERVE" ] || [ "$REC/serve/serve.sh" -nt "$DSERVE" ]; then
    python3 "$ROOT/tools/patch_diffbot_serve.py" "$REC" "$DSERVE"
fi
echo "[serve] pack $PACK (2.9 bpw EXL3 on diffbot's stack), plan $PLAN, kernels $XMOE_EXT + $EXL3_DO_DIR, auth $([ -n "$API_KEY" ] && echo 'Bearer token' || echo none)"
exec bash "$DSERVE"
