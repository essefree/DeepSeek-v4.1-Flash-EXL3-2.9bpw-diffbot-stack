# sitecustomize for the 2.9 bpw pack on diffbot's stack: diffbot's own sitecustomize first (their
# DS41_* hooks, mounted by their serve.sh at /opt/recipe), then the hooks this quant needs.
#
# Mounted at /usr/lib/python3.12/sitecustomize.py by the patched copy of diffbot's serve.sh
# (tools/patch_diffbot_serve.py), so it runs in every vLLM process (API, EngineCore, workers).
import importlib.abc
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
#
# Applied when vLLM imports the model module, not by importing it here: importing vLLM's model code
# from sitecustomize would run it before the vllm-exl3 plugin (and diffbot's patches of it) are
# registered, and `from X import y` bindings made at that point would miss their patches.
_MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")


def _ds29_patch_model_module(mod) -> None:
    patched = []
    for cls_name in dir(mod):
        cls = getattr(mod, cls_name)
        if not isinstance(cls, type) or "_pad_shared_expert_weight" not in vars(cls):
            continue
        orig = vars(cls)["_pad_shared_expert_weight"]
        if getattr(orig, "_ds29_wrapped", False):
            continue
        fn = orig.__func__ if isinstance(orig, (staticmethod, classmethod)) else orig

        # called as self._pad_shared_expert_weight(self.quant_config, name, loaded_weight)
        def wrapped(*args, _fn=fn, **kwargs):
            name = next((a for a in args if isinstance(a, str)), kwargs.get("name", ""))
            weight = next((a for a in args if hasattr(a, "dim") and callable(a.dim)), kwargs.get("loaded_weight"))
            if weight is not None and (name.endswith(_EXL3_SUFFIXES) or weight.dim() != 2):
                return weight
            return _fn(*args, **kwargs)

        wrapped._ds29_wrapped = True
        if isinstance(orig, staticmethod):
            setattr(cls, "_pad_shared_expert_weight", staticmethod(wrapped))
        elif isinstance(orig, classmethod):
            setattr(cls, "_pad_shared_expert_weight", classmethod(wrapped))
        else:
            setattr(cls, "_pad_shared_expert_weight", wrapped)
        patched.append(cls_name)
    print(f"[sitecustomize-2.9bpw] EXL3 shared experts skip block-FP8 padding: {patched or 'no class found'}",
          file=sys.stderr)


# vllm-exl3 dense linears on a ReplicatedLinear (the DSA indexer's wq_b: 1280 -> 4096, EXL3 K=5 in this
# quant). The plugin's _resolve_tp_geometry() takes the layer's tp_rank/tp_size, which vLLM's LinearBase
# sets from the global TP size for every linear that is not disable_tp -- a ReplicatedLinear included,
# although vLLM never shards it. With TP=2 the plugin then narrowed the 4096-wide svh to 2048 for a
# 4096-wide parameter ("EXL3 linear load shape mismatch shard=0 suffix=svh: dest (4096,) != loaded (2048,)").
# diffbot's pack has no EXL3 dense modules, so their stack never met this. Replicated = TP geometry (0, 1).
_PLUGIN_MODULE = "vllm_exl3.exl3"


def _ds29_patch_plugin_module(mod) -> None:
    orig = getattr(mod, "_resolve_tp_geometry", None)
    if orig is None or getattr(orig, "_ds29_wrapped", False):
        print(f"[sitecustomize-2.9bpw] vllm-exl3: _resolve_tp_geometry {'already patched' if orig else 'not found'}",
              file=sys.stderr)
        return
    try:
        from vllm.model_executor.layers.linear import ReplicatedLinear
    except Exception as exc:  # pragma: no cover
        print(f"[sitecustomize-2.9bpw] cannot import ReplicatedLinear: {exc!r}", file=sys.stderr)
        return

    def resolve_tp_geometry(*owners, _orig=orig):
        if any(isinstance(o, ReplicatedLinear) for o in owners):
            return 0, 1
        return _orig(*owners)

    resolve_tp_geometry._ds29_wrapped = True
    mod._resolve_tp_geometry = resolve_tp_geometry
    print("[sitecustomize-2.9bpw] vllm-exl3: ReplicatedLinear EXL3 modules load unsharded (TP geometry 0/1)",
          file=sys.stderr)


_POST_IMPORT = {_MODEL_MODULE: _ds29_patch_model_module, _PLUGIN_MODULE: _ds29_patch_plugin_module}


class _Ds29PostImport(importlib.abc.MetaPathFinder):
    """Runs the matching _POST_IMPORT hook right after one of those modules is executed."""

    def find_spec(self, fullname, path=None, target=None):
        hook = _POST_IMPORT.get(fullname)
        if hook is None:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        loader, orig_exec = spec.loader, spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _hook=hook):
            _orig(module)
            _hook(module)

        loader.exec_module = exec_module  # SourceFileLoader instances are per-module
        return spec


for _name, _hook in _POST_IMPORT.items():
    if _name in sys.modules:
        _hook(sys.modules[_name])
if not any(isinstance(f, _Ds29PostImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _Ds29PostImport())
