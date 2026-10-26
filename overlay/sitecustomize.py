# sitecustomize for the 2.9 bpw pack on diffbot's stack: diffbot's own sitecustomize first (their
# DS41_* hooks, mounted by their serve.sh at /opt/recipe), then the hooks this quant needs.
#
# Mounted at /usr/lib/python3.12/sitecustomize.py by the patched copy of diffbot's serve.sh
# (tools/patch_diffbot_serve.py), so it runs in every vLLM process (API, EngineCore, workers).
import os
import sys

_DIFFBOT = os.environ.get("DS29_DIFFBOT_SITECUSTOMIZE", "/opt/recipe/patches/vllm/sitecustomize.py")
try:
    with open(_DIFFBOT) as _f:
        exec(compile(_f.read(), _DIFFBOT, "exec"), {"__name__": "sitecustomize_diffbot", "__file__": _DIFFBOT})
except FileNotFoundError:
    print(f"[sitecustomize-2.9bpw] WARNING: diffbot's sitecustomize not found at {_DIFFBOT}", file=sys.stderr)


# EXL3 shared experts. DeepseekV4Model.load_weights pads every `.shared_experts.` tensor to a
# TP-uniform block count when the quant config carries weight_block_size (it does: diffbot's
# sitecustomize copies it onto Exl3Config for the FP8 delegate). That padding assumes a 2-D
# block-FP8 matrix; the packed EXL3 tensors of this quant's shared experts (trellis [k/16, n/16, K*16],
# suh/svh [n], the mul1 marker, a 0-dim int) are not, and vllm-exl3 pads its own matrices. The
# block-FP8 shared down-proj that convert_natives.py produced is a no-op for the padding anyway
# (2304 intermediate = 36 x 64).
def _ds29_skip_pad_for_exl3() -> None:
    try:
        import vllm.models.deepseek_v41.nvidia.model as _m
    except Exception as exc:  # pragma: no cover
        print(f"[sitecustomize-2.9bpw] cannot import deepseek_v41 model: {exc!r}", file=sys.stderr)
        return
    suffixes = (".trellis", ".suh", ".svh", ".mul1", ".mcg")
    patched = []
    for cls_name in dir(_m):
        cls = getattr(_m, cls_name)
        orig = getattr(cls, "_pad_shared_expert_weight", None) if isinstance(cls, type) else None
        if orig is None or getattr(orig, "_ds29_wrapped", False):
            continue
        fn = orig.__func__ if isinstance(orig, staticmethod) else orig

        def wrapped(quant_config, name, loaded_weight, _fn=fn):
            if name.endswith(suffixes) or loaded_weight.dim() != 2:
                return loaded_weight
            return _fn(quant_config, name, loaded_weight)

        wrapped._ds29_wrapped = True
        setattr(cls, "_pad_shared_expert_weight", staticmethod(wrapped))
        patched.append(cls_name)
    print(f"[sitecustomize-2.9bpw] EXL3 shared experts skip block-FP8 padding: {patched or 'no class found'}",
          file=sys.stderr)


_ds29_skip_pad_for_exl3()
