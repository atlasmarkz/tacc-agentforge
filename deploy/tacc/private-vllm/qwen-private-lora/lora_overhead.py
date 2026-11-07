"""Where does the ~23% single-stream decode slowdown with a (zero) LoRA adapter come from?

Three servers, same flags as the `graphs` baseline (compiled + CUDA graphs, 4K context):
  nolora - no --enable-lora (reference)
  lora1  - --enable-lora, --max-loras 1
  lora5  - --enable-lora, --max-loras 5
On each, decode speed (256 tokens, ignore_eos, temperature 0) for the base model and for zero-B adapters that
cover different module sets, interleaved, 3 repeats each; then 4 concurrent requests per target.
Adapters (rank 8, B = 0, so outputs equal the base): all_zero, mlp_zero, full_zero, lin_zero.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import lora_e4b as e4

PORT = e4.PORT
PROMPT = "Write a long, detailed essay about the history of computing."
ADAPTERS = {  # name -> plan for make_adapter
    "all_zero": ({"mlp": e4.MLP, "full": e4.FULL, "lin": e4.LIN}, e4.MLP + e4.FULL + e4.LIN),
    "mlp_zero": ({"mlp": e4.MLP, "full": [], "lin": []}, e4.MLP),
    "full_zero": ({"mlp": [], "full": e4.FULL, "lin": []}, e4.FULL),
    "lin_zero": ({"mlp": [], "full": [], "lin": e4.LIN}, e4.LIN),
}


def one(model: str, n_tokens: int = 256) -> tuple:
    t0 = time.time()
    r = e4.post("/v1/completions", {"model": model, "prompt": PROMPT, "max_tokens": n_tokens,
                                    "temperature": 0, "ignore_eos": True})
    return r["usage"]["completion_tokens"], time.time() - t0


def concurrent(model: str, n: int = 4, n_tokens: int = 128) -> float:
    toks = []

    def worker() -> None:
        toks.append(one(model, n_tokens)[0])

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return round(sum(toks) / (time.time() - t0), 1)


def run_server(name: str, lora_slots: int, work: Path, out: Path, model_path: str, timeout_s: int) -> dict:
    res: dict = {}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--served-model-name", "qwen3.8-27b",
           "--host", "127.0.0.1", "--port", str(PORT), "--tensor-parallel-size", "2",
           "--safetensors-load-strategy", "prefetch", "--language-model-only", "--dtype", "bfloat16",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--max-num-seqs", "8",
           "--max-num-batched-tokens", "2048", "--disable-custom-all-reduce"]
    models = ["qwen3.8-27b"]
    if lora_slots:
        cmd += ["--enable-lora", "--max-lora-rank", str(e4.RANK), "--max-loras", str(lora_slots),
                "--lora-modules", *[f"{n}={work / n}" for n in ADAPTERS]]
        models += list(ADAPTERS)
    env = dict(os.environ, NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    log = open(out / f"server-{name}.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        while True:
            if proc.poll() is not None:
                res.update(status="server_exited", exit_code=proc.returncode)
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
        for m in models:  # warm each target once (loads adapters, triggers kernel JIT)
            one(m, 16)
        single: dict = {m: [] for m in models}
        for _ in range(3):
            for m in models:
                toks, dt = one(m)
                single[m].append(round(toks / dt, 1))
        res["single_stream_tok_per_s"] = single
        res["concurrent4_tok_per_s"] = {m: concurrent(m) for m in models}
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
    sh = e4.Shapes(model_path)
    for i, (n, (plan, targets)) in enumerate(ADAPTERS.items()):
        e4.make_adapter(work / n, sh, plan, 0.0, i + 1, targets)
    report: dict = {"model_path": model_path, "rank": e4.RANK, "prompt": PROMPT}
    t_start = time.time()
    for name, slots in [("nolora", 0), ("lora5", 5), ("lora1", 1)]:
        if time.time() - t_start > 2400:
            report[name] = {"status": "skipped_time"}
            continue
        report[name] = run_server(name, slots, work, out, model_path, 1200)
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
