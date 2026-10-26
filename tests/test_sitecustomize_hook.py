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
    os.environ["DS29_DIFFBOT_SITECUSTOMIZE"] = str(tmp / "missing-diffbot-sitecustomize.py")
    for m in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
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
    print("test_sitecustomize_hook: OK (hook applied on import; EXL3 tensors untouched; 2-D tensors padded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
