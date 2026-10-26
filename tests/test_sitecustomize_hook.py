#!/usr/bin/env python3
"""overlay/sitecustomize.py: the shared-expert padding hook applies when vLLM's model module is imported
(a fake vllm.models.deepseek_v41.nvidia.model stands in), is signature-agnostic, and leaves 2-D
block-FP8 tensors to the original. No torch needed."""
from __future__ import annotations

import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ds29-site-"))
    pkg = tmp / "vllm" / "models" / "deepseek_v41" / "nvidia"
    pkg.mkdir(parents=True)
    for d in (tmp / "vllm", tmp / "vllm" / "models", tmp / "vllm" / "models" / "deepseek_v41", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "model.py").write_text(
        "class DeepseekV4Model:\n"
        "    @staticmethod\n"
        "    def _pad_shared_expert_weight(quant_config, name, loaded_weight):\n"
        "        loaded_weight.shape = (loaded_weight.shape[0] + 1, loaded_weight.shape[1])  # 'padded'\n"
        "        return loaded_weight\n"
        "class Other:\n"
        "    pass\n")
    # fake vLLM linear classes + fake vllm-exl3 plugin for the TP-geometry hook
    lin = tmp / "vllm" / "model_executor" / "layers"
    lin.mkdir(parents=True)
    (tmp / "vllm" / "model_executor" / "__init__.py").write_text("")
    (lin / "__init__.py").write_text("")
    (lin / "linear.py").write_text(
        "class LinearBase:\n"
        "    def __init__(self, tp_rank=1, tp_size=2):\n"
        "        self.tp_rank, self.tp_size = tp_rank, tp_size\n"
        "class ReplicatedLinear(LinearBase):\n"
        "    pass\n"
        "class ColumnParallelLinear(LinearBase):\n"
        "    pass\n")
    (tmp / "vllm_exl3").mkdir()
    (tmp / "vllm_exl3" / "__init__.py").write_text("")
    (tmp / "vllm_exl3" / "exl3.py").write_text(
        "def _resolve_tp_geometry(*owners):\n"
        "    for o in owners:\n"
        "        if o is not None and hasattr(o, 'tp_rank'):\n"
        "            return int(o.tp_rank), int(o.tp_size)\n"
        "    return 0, 1\n"
        "def loader_geometry(layer, param=None):\n"
        "    return _resolve_tp_geometry(layer, param)  # global lookup at call time, as the real loader does\n")
    os.environ["DS29_DIFFBOT_SITECUSTOMIZE"] = str(tmp / "missing-diffbot-sitecustomize.py")
    for m in [m for m in sys.modules if m in ("vllm", "vllm_exl3") or m.startswith(("vllm.", "vllm_exl3."))]:
        del sys.modules[m]
    sys.path.insert(0, str(tmp))
    runpy.run_path(str(ROOT / "overlay" / "sitecustomize.py"), run_name="sitecustomize")
    import vllm.models.deepseek_v41.nvidia.model as m  # noqa: E402  (fake, from tmp)

    cls = m.DeepseekV4Model
    assert getattr(vars(cls)["_pad_shared_expert_weight"].__func__, "_ds29_wrapped", False), "hook not applied"
    inst = cls()
    marker = FakeTensor(())
    assert inst._pad_shared_expert_weight(None, "layers.3.ffn.shared_experts.w1.mul1", marker) is marker
    assert marker.shape == ()
    trellis = FakeTensor((320, 144, 80))
    assert inst._pad_shared_expert_weight(None, "layers.3.ffn.shared_experts.w1.trellis", trellis) is trellis
    suh = FakeTensor((5120,))
    assert inst._pad_shared_expert_weight(None, "layers.3.ffn.shared_experts.w1.suh", suh).shape == (5120,)
    fp8 = FakeTensor((2304, 5120))
    assert inst._pad_shared_expert_weight(None, "layers.12.ffn.shared_experts.w2.weight", fp8).shape == (2305, 5120)
    assert inst._pad_shared_expert_weight(None, "layers.12.ffn.shared_experts.w2.weight_scale_inv",
                                          FakeTensor((72, 160))).shape == (73, 160)
    import vllm_exl3.exl3 as plug  # noqa: E402  (fake, from tmp)
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear  # noqa: E402

    assert getattr(plug._resolve_tp_geometry, "_ds29_wrapped", False), "plugin hook not applied"
    assert plug.loader_geometry(ReplicatedLinear(tp_rank=1, tp_size=2)) == (0, 1), "replicated must not be sharded"
    assert plug.loader_geometry(ColumnParallelLinear(tp_rank=1, tp_size=2)) == (1, 2), "column-parallel keeps TP"
    assert plug.loader_geometry(ColumnParallelLinear(tp_rank=1, tp_size=2), None) == (1, 2)
    print("test_sitecustomize_hook: OK (padding hook applied on import; EXL3 tensors untouched; 2-D tensors padded; "
          "ReplicatedLinear TP geometry (0, 1))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
