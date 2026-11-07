"""E4: can the private vLLM apply a PEFT-format LoRA to the course Qwen3.8-27B INT4 model?

Writes three dummy adapters on the MLP projections of all 64 layers (rank 8):
  zero  - lora_B = 0, so outputs must equal the base model's
  small - lora_B ~ N(0, 0.01)
  large - lora_B ~ N(0, 0.1)
starts one server with --enable-lora, then compares greedy completions. The base model is queried
through the same server, so batch composition is identical for all four.
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
from safetensors.torch import save_file

PORT = 8011
HIDDEN, INTER, LAYERS, RANK = 5120, 17408, 64, 8
MLP = {"gate_proj": (HIDDEN, INTER), "up_proj": (HIDDEN, INTER), "down_proj": (INTER, HIDDEN)}
PROMPTS = ["The capital of France is", "Three prime numbers greater than 10 are",
           "Translate to French: good morning, how are you?"]


def make_adapter(path: Path, model_path: str, b_std: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    t = {}
    for i in range(LAYERS):
        for name, (fin, fout) in MLP.items():
            k = f"base_model.model.model.language_model.layers.{i}.mlp.{name}"
            t[f"{k}.lora_A.weight"] = (torch.randn(RANK, fin, generator=g) * 0.02).contiguous()
            b = torch.zeros(fout, RANK) if b_std == 0 else torch.randn(fout, RANK, generator=g) * b_std
            t[f"{k}.lora_B.weight"] = b.contiguous()
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": 16,
           "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "target_modules": list(MLP), "base_model_name_or_path": model_path}
    (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    model_path = os.environ["QWEN_MODEL_PATH"]
    report: dict = {"model_path": model_path, "rank": RANK, "targets": list(MLP), "layers": LAYERS}

    adapters = {"zero": 0.0, "small": 0.01, "large": 0.1}
    for n, std in adapters.items():
        make_adapter(work / n, model_path, std, seed=len(n))
    report["adapter_dir_bytes"] = {n: sum(f.stat().st_size for f in (work / n).iterdir()) for n in adapters}

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--served-model-name", "qwen3.8-27b",
           "--host", "127.0.0.1", "--port", str(PORT), "--tensor-parallel-size", "2",
           "--safetensors-load-strategy", "prefetch", "--language-model-only", "--dtype", "bfloat16",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--max-num-seqs", "8",
           "--max-num-batched-tokens", "2048", "--disable-custom-all-reduce", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", str(RANK), "--max-loras", "3",
           "--lora-modules", *[f"{n}={work / n}" for n in adapters]]
    env = dict(os.environ, NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    report["command"] = cmd
    log = open(out / "server.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    def save() -> None:
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        while True:
            if proc.poll() is not None:
                report["status"] = "server_exited"
                report["exit_code"] = proc.returncode
                return
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except Exception:
                if time.time() - t0 > 1500:
                    report["status"] = "startup_timeout"
                    return
                time.sleep(5)
        report["startup_s"] = round(time.time() - t0, 1)
        models = [m["id"] for m in post_get("/v1/models")["data"]]
        report["served_models"] = models
        outputs: dict = {}
        for name in ["qwen3.8-27b", *adapters]:
            outputs[name] = []
            for p in PROMPTS:
                try:
                    r = post("/v1/completions", {"model": name, "prompt": p, "max_tokens": 40,
                                                 "temperature": 0})
                    outputs[name].append(r["choices"][0]["text"])
                except Exception as e:  # keep the error text; a failed adapter is a result
                    outputs[name].append(f"ERROR: {e!r}")
        report["outputs"] = outputs
        base = outputs["qwen3.8-27b"]
        report["equals_base"] = {n: outputs[n] == base for n in adapters}
        report["status"] = "ok"
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        save()


def post_get(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30) as r:
        return json.load(r)


if __name__ == "__main__":
    main()
