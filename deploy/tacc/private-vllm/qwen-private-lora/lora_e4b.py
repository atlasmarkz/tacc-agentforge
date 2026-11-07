"""E4b: LoRA on attention and linear-attention projections, then with CUDA graphs on.

Adapters (PEFT layout, rank 8, shapes read from the checkpoint):
  all_zero - B = 0 on MLP + full attention + linear attention (must equal base)
  full_b   - B ~ N(0, 0.05) on self_attn q/k/v/o_proj only (full-attention layers)
  lin_b    - B ~ N(0, 0.05) on linear_attn in_proj_qkv/in_proj_z/out_proj only (linear-attention layers)
Phase 1: --enforce-eager.  Phase 2: compiled + CUDA graphs (the course default), plus a single-stream
decode-speed check of base vs the zero adapter.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PORT = 8012
RANK = 8
PROMPTS = ["The capital of France is", "Three prime numbers greater than 10 are",
           "Translate to French: good morning, how are you?"]
MLP = ["gate_proj", "up_proj", "down_proj"]
FULL = ["q_proj", "k_proj", "v_proj", "o_proj"]
LIN = ["in_proj_qkv", "in_proj_z", "out_proj"]


class Shapes:
    """(in_features, out_features) of each Linear, read from safetensors headers."""

    def __init__(self, model_path: str):
        self.dir = Path(model_path)
        self.wm = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        cfg = json.load(open(self.dir / "config.json"))
        self.types = cfg.get("text_config", cfg)["layer_types"]

    def _shape(self, name: str) -> list:
        with safe_open(str(self.dir / self.wm[name]), "pt") as f:
            return list(f.get_slice(name).get_shape())

    def linear(self, base: str) -> tuple:
        if f"{base}.weight_scale" in self.wm:      # quantized: scale is (out, in / group_size)
            out, groups = self._shape(f"{base}.weight_scale")
            return groups * 32, out
        out, fin = self._shape(f"{base}.weight")
        return fin, out


def make_adapter(path: Path, sh: Shapes, plan: dict, b_std: float, seed: int, targets: list) -> dict:
    g = torch.Generator().manual_seed(seed)
    t, used = {}, {}
    for i, kind in enumerate(sh.types):
        mods = [("mlp", m) for m in plan["mlp"]]
        if kind == "full_attention":
            mods += [("self_attn", m) for m in plan["full"]]
        if kind == "linear_attention":
            mods += [("linear_attn", m) for m in plan["lin"]]
        for blk, m in mods:
            base = f"model.language_model.layers.{i}.{blk}.{m}"
            fin, fout = sh.linear(base)
            used.setdefault(f"{blk}.{m}", [fin, fout])
            k = f"base_model.model.{base}"
            t[f"{k}.lora_A.weight"] = (torch.randn(RANK, fin, generator=g) * 0.02).contiguous()
            b = torch.zeros(fout, RANK) if b_std == 0 else torch.randn(fout, RANK, generator=g) * b_std
            t[f"{k}.lora_B.weight"] = b.contiguous()
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": 16, "lora_dropout": 0.0,
           "bias": "none", "fan_in_fan_out": False, "target_modules": targets,
           "base_model_name_or_path": str(sh.dir)}
    (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return used


def post(path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30) as r:
        return json.load(r)


def run_phase(name: str, extra: list, adapters: dict, work: Path, out: Path, model_path: str,
              timeout_s: int, speed: bool) -> dict:
    res: dict = {}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--served-model-name", "qwen3.8-27b",
           "--host", "127.0.0.1", "--port", str(PORT), "--tensor-parallel-size", "2",
           "--safetensors-load-strategy", "prefetch", "--language-model-only", "--dtype", "bfloat16",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--max-num-seqs", "8",
           "--max-num-batched-tokens", "2048", "--disable-custom-all-reduce", *extra,
           "--enable-lora", "--max-lora-rank", str(RANK), "--max-loras", "5",
           "--lora-modules", *[f"{n}={work / n}" for n in adapters]]
    res["command"] = cmd
    env = dict(os.environ, NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    log = open(out / f"server-{name}.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        while True:
            if proc.poll() is not None:
                res["status"], res["exit_code"] = "server_exited", proc.returncode
                return res
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except Exception:
                if time.time() - t0 > timeout_s:
                    res["status"] = "startup_timeout"
                    return res
                time.sleep(5)
        res["startup_s"] = round(time.time() - t0, 1)
        res["served_models"] = [m["id"] for m in get("/v1/models")["data"]]
        outs: dict = {}
        lps: dict = {}
        for n in ["qwen3.8-27b", *adapters]:
            outs[n] = []
            for p in PROMPTS:
                try:
                    r = post("/v1/completions", {"model": n, "prompt": p, "max_tokens": 40,
                                                 "temperature": 0, "logprobs": 1})
                    outs[n].append(r["choices"][0]["text"])
                    lps.setdefault(n, []).append(r["choices"][0]["logprobs"]["token_logprobs"])
                except Exception as e:
                    outs[n].append(f"ERROR: {e!r}")
        res["outputs"] = outs
        base = outs["qwen3.8-27b"]
        res["equals_base"] = {n: outs[n] == base for n in adapters}
        res["max_abs_logprob_diff_vs_base"] = {
            n: max((abs(a - b) for x, y in zip(lps[n], lps["qwen3.8-27b"])
                    for a, b in zip(x, y) if a is not None and b is not None), default=None)
            for n in adapters if n in lps}
        res["n_differ_from_base"] = {n: sum(x != y for x, y in zip(outs[n], base)) for n in adapters}
        if speed:
            sp: dict = {}
            for n in ["qwen3.8-27b", "all_zero", "qwen3.8-27b", "all_zero"]:
                t1 = time.time()
                r = post("/v1/completions", {"model": n, "prompt": PROMPTS[0], "max_tokens": 128,
                                             "temperature": 0, "ignore_eos": True})
                dt = time.time() - t1
                sp.setdefault(n, []).append(round(r["usage"]["completion_tokens"] / dt, 1))
            res["decode_tok_per_s_single_stream"] = sp
        res["status"] = "ok"
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    model_path = os.environ["QWEN_MODEL_PATH"]
    sh = Shapes(model_path)
    report: dict = {"model_path": model_path, "rank": RANK,
                    "layer_types_count": {t: sh.types.count(t) for t in set(sh.types)}}

    def save() -> None:
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["module_shapes_in_out"] = {}
    report["module_shapes_in_out"].update(make_adapter(
        work / "all_zero", sh, {"mlp": MLP, "full": FULL, "lin": LIN}, 0.0, 1, MLP + FULL + LIN))
    report["module_shapes_in_out"].update(make_adapter(
        work / "full_b", sh, {"mlp": [], "full": FULL, "lin": []}, 0.05, 2, FULL))
    report["module_shapes_in_out"].update(make_adapter(
        work / "lin_b", sh, {"mlp": [], "full": [], "lin": LIN}, 0.05, 3, LIN))
    report["module_shapes_in_out"].update(make_adapter(
        work / "full_big", sh, {"mlp": [], "full": FULL, "lin": []}, 0.5, 4, FULL))
    report["module_shapes_in_out"].update(make_adapter(
        work / "lin_big", sh, {"mlp": [], "full": [], "lin": LIN}, 0.5, 5, LIN))
    keep = os.environ.get("E4_ADAPTERS")
    adapters = {n: 0 for n in ["all_zero", "full_b", "lin_b", "full_big", "lin_big"]
                if not keep or n in keep.split(",")}
    save()
    try:
        report["eager"] = run_phase("eager", ["--enforce-eager"], adapters, work, out, model_path, 1500, False)
        save()
        if report["eager"].get("status") == "ok" and os.environ.get("E4_COMPILED", "1") == "1":
            report["compiled"] = run_phase("compiled", [], adapters, work, out, model_path, 1800, True)
    finally:
        save()


if __name__ == "__main__":
    main()
