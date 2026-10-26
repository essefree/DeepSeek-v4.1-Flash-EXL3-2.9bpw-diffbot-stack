#!/usr/bin/env bash
# Serve the assembled pack through diffbot's recipe/serve/serve.sh with the settings this quant needs.
#
# Everything else (image, patches, mounts, API, CPU_EXPERTS, 1M context, DSpark k=5, warmup) is theirs;
# their script documents every knob and all of them can still be set in the environment.
#
#   API_KEY=change-me ./serve.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$ROOT/.env" ] || { echo "no .env (cp env.example .env)"; exit 1; }
set -a; source "$ROOT/.env"; set +a
: "${API_KEY:?set API_KEY}"
: "${DIFFBOT:?}" "${PACK:?}"
REC=$DIFFBOT/recipe
[ -f "$PACK/config.json" ] || { echo "no $PACK/config.json: run ./prepare_pack.py first"; exit 1; }
[ -f "$PACK/PACK_INFO.json" ] || { echo "$PACK was not assembled by prepare_pack.py (no PACK_INFO.json)"; exit 1; }
PLAN_NAME=${PLAN_NAME:-plan-2.9bpw-${DIFFBOT_PLAN:-plan-1017-agent30-c512.json}}
[ -f "$REC/plans/$PLAN_NAME" ] || { echo "no $REC/plans/$PLAN_NAME: run ./prepare_pack.py (it derives and installs the plan)"; exit 1; }
[ -d "$REC/kernels/build/xmoe_k2" ] || { echo "no $REC/kernels/build/xmoe_k2: run kernels/build.sh"; exit 1; }
[ -d "$REC/kernels/build/k2/xmoe_do_i8" ] || { echo "no $REC/kernels/build/k2/xmoe_do_i8: run kernels/build.sh"; exit 1; }

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
echo "[serve] pack $PACK (2.9 bpw EXL3 on diffbot's stack), plan $PLAN, kernels $XMOE_EXT + $EXL3_DO_DIR"
exec bash "$REC/serve/serve.sh"
