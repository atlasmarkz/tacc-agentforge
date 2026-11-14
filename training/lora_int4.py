"""LoRA on the course Qwen3.8-27B AWQ-INT4 checkpoint, in HuggingFace, without peft (Task 7).

From the feasibility test (docs/experiments/2026-10-28-27b-lora-feasibility/README.md, job 27833):
- AutoModelForImageTextToText loads the checkpoint correctly; AutoModelForCausalLM silently leaves the weights
  unloaded (key names model.language_model.*).
- compressed-tensors 0.17 attaches a decompression hook that crashes on this checkpoint; it is removed and the
  INT4 weights are unpacked by int4_dequant (group size 32, asymmetric, packed 8 per int32).
- peft cannot wrap these quantized layers (no .weight), so a small LoRA module wraps them instead.

Adapters are exported in the peft layout that vLLM serves on this model (docs/experiments/
2026-11-06-private-vllm-feasibility/PLAN.md, E4 and E4b, jobs 28345 and 28355): keys
"base_model.model.<module>.lora_{A,B}.weight" with <module> = model.language_model.layers.N.<block>.<proj>.
A key one "model." short loaded nothing in vLLM, silently (jobs 28347-28352), so the prefix is checked.
Torch and transformers are imported inside functions: this module imports without them (tests, local tools).
"""
from __future__ import annotations

import json
from pathlib import Path

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj")
PEFT_PREFIX = "base_model.model."
MODULE_PREFIX = "model.language_model.layers."


def peft_key(module_name: str, which: str) -> str:
    """'model.language_model.layers.3.mlp.gate_proj', 'A' -> peft tensor name served by vLLM."""
    if not module_name.startswith(MODULE_PREFIX):
        raise ValueError(f"unexpected module name {module_name!r}: vLLM would ignore this adapter weight")
    return f"{PEFT_PREFIX}{module_name}.lora_{which}.weight"


def module_from_peft_key(key: str) -> tuple[str, str]:
    if not key.startswith(PEFT_PREFIX + MODULE_PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"not an adapter key of this model: {key!r}")
    body = key[len(PEFT_PREFIX):-len(".weight")]
    module, _, which = body.rpartition(".lora_")
    return module, which


def adapter_config(r: int, alpha: int, targets, base_model: str) -> dict:
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
            "bias": "none", "fan_in_fan_out": False, "target_modules": sorted(set(targets)),
            "base_model_name_or_path": str(base_model)}


def load_model(model_path: str, max_memory_gib: int = 21):
    import torch
    import transformers
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{max_memory_gib}GiB", 1: f"{max_memory_gib}GiB"})
    removed = 0
    for mod in model.modules():
        for hooks in (mod._forward_hooks, mod._forward_pre_hooks):
            for key, hook in list(hooks.items()):
                if "decompress" in repr(hook):
                    del hooks[key]
                    removed += 1
    patched = 0
    import types
    for mod in model.modules():
        if hasattr(mod, "weight_packed"):
            mod.forward = types.MethodType(_quantized_forward, mod)
            patched += 1
    return model, {"removed_hooks": removed, "quantized_modules": patched}


def int4_dequant(mod):
    """Unpack a compressed-tensors pack-quantized INT4 group-quantized Linear into a bf16 weight
    (layout and reasoning: see the feasibility README; verified there by a correct greedy completion)."""
    import torch
    packed, scale = mod.weight_packed, mod.weight_scale
    zp = getattr(mod, "weight_zero_point", None)
    out_f, in_f = packed.shape[0], packed.shape[1] * 8
    groups = scale.shape[1]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(out_f, in_f).to(torch.int16) - 8
    q = q.view(out_f, groups, in_f // groups)
    if zp is not None:
        z = ((zp.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(out_f, groups).to(torch.int16) - 8
        q = q - z.unsqueeze(-1)
    return (q.to(scale.dtype) * scale.unsqueeze(-1)).reshape(out_f, in_f).to(torch.bfloat16)


def _quantized_forward(self, x):
    import torch
    with torch.no_grad():
        w = int4_dequant(self)
    return torch.nn.functional.linear(x, w.to(x.dtype), getattr(self, "bias", None))


def make_lora_class():
    import torch

    class LoRALinear(torch.nn.Module):
        """Trainable low-rank update (fp32) on top of a frozen, possibly quantized, layer."""

        def __init__(self, base, r: int, alpha: int):
            super().__init__()
            self.base = base
            device = (list(base.parameters()) + list(base.buffers()))[0].device
            self.r, self.scale = r, alpha / r
            self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=torch.float32))
            self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        def forward(self, x):
            out = self.base(x)
            update = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
            return out + (update * self.scale).to(out.dtype)

    return LoRALinear


def add_lora(model, r: int, alpha: int, targets=LORA_TARGETS) -> list[str]:
    """Freeze the model and wrap every target Linear of the language model. Returns wrapped module names."""
    import torch
    LoRALinear = make_lora_class()
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if child_name in targets and isinstance(child, torch.nn.Linear) and full.startswith(MODULE_PREFIX):
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                wrapped.append(full)
    if not wrapped:
        raise RuntimeError("no target modules found: check the loader class and module names")
    return wrapped


def lora_modules(model) -> dict:
    return {name: mod for name, mod in model.named_modules() if hasattr(mod, "lora_A") and hasattr(mod, "lora_B")}


def export_peft(model, out_dir: Path, r: int, alpha: int, base_model: str, extra_meta: dict | None = None) -> dict:
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, targets = {}, set()
    for name, mod in lora_modules(model).items():
        tensors[peft_key(name, "A")] = mod.lora_A.detach().float().cpu().contiguous()
        tensors[peft_key(name, "B")] = mod.lora_B.detach().float().cpu().contiguous()
        targets.add(name.rsplit(".", 1)[-1])
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    cfg = adapter_config(r, alpha, targets, base_model)
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if extra_meta is not None:
        (out_dir / "training_meta.json").write_text(json.dumps(extra_meta, indent=2), encoding="utf-8")
    return {"tensors": len(tensors), "modules": len(tensors) // 2, "targets": sorted(targets)}


def import_peft(model, adapter_dir: Path) -> dict:
    """Load an exported adapter into a model that already has LoRA modules (round-trip check)."""
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    mods = lora_modules(model)
    loaded = 0
    for key, value in tensors.items():
        module, which = module_from_peft_key(key)
        target = getattr(mods[module], f"lora_{which}")
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(value.shape)} vs {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(value.to(target.device, target.dtype))
        loaded += 1
    missing = 2 * len(mods) - loaded
    return {"loaded": loaded, "missing": missing, "r": cfg["r"], "lora_alpha": cfg["lora_alpha"]}
