"""E6: per-request timing breakdown (queue, prefill, decode) from vLLM's own metrics, next to client timings.

vLLM already records per-request phase times as Prometheus histograms on /metrics:
  vllm:request_queue_time_seconds, request_prefill_time_seconds, request_decode_time_seconds,
  request_inference_time_seconds, e2e_request_latency_seconds, time_to_first_token_seconds.
No source change is needed: we scrape _sum/_count before and after each workload cell and report the mean of each
phase for the requests in that cell, next to the client-measured TTFT and end-to-end time (streaming client).

Two servers (graphs baseline flags, compiled + CUDA graphs): without and with --enable-prefix-caching, so the
effect of caching shows up in the prefill phase specifically.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from benchmark import run_benchmark
from experiments import PORT, wait_ready, workload_prompts, stop_server

WANTED = re.compile(r"^vllm:(request_queue_time_seconds|request_prefill_time_seconds|"
                    r"request_decode_time_seconds|request_inference_time_seconds|"
                    r"e2e_request_latency_seconds|time_to_first_token_seconds|"
                    r"request_time_per_output_token_seconds|inter_token_latency_seconds)_(sum|count)\{")
CELLS = [(c, k) for c in (1, 4, 8) for k in ("distinct", "shared_prefix")]
BASE_ARGS = ["--safetensors-load-strategy", "prefetch", "--language-model-only", "--dtype", "bfloat16",
             "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--max-num-seqs", "8",
             "--max-num-batched-tokens", "2048", "--reasoning-parser", "qwen3", "--disable-custom-all-reduce"]


def scrape() -> dict:
    text = urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    acc: dict = {}
    for line in text.splitlines():
        m = WANTED.match(line)
        if m:
            key = f"{m.group(1)}_{m.group(2)}"
            acc[key] = acc.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return acc


def phase_means(before: dict, after: dict) -> dict:
    out = {}
    for k in {k[:-4] for k in after if k.endswith("_sum")}:
        n = after.get(k + "_count", 0) - before.get(k + "_count", 0)
        s = after.get(k + "_sum", 0) - before.get(k + "_sum", 0)
        out[k] = {"requests": int(n), "mean_ms": round(1000 * s / n, 1) if n else None}
    return out


def run_server(name: str, extra: list, model_path: str, out: Path) -> dict:
    res: dict = {"extra_args": extra}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
           "--served-model-name", "qwen3.8-27b", "--host", "127.0.0.1", "--port", str(PORT),
           "--tensor-parallel-size", "2", *BASE_ARGS, *extra]
    env = dict(os.environ, NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    log = open(out / f"server-{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        ready = wait_ready(proc, 1500)
        if ready is None:
            res["status"] = "not_ready"
            return res
        res["startup_s"] = ready
        res["cells"] = {}
        for conc, kind in CELLS:
            before = scrape()
            run = run_benchmark(f"http://127.0.0.1:{PORT}/v1", workload_prompts(kind), name,
                                model="qwen3.8-27b", repeats=2, concurrency=conc, max_tokens=128)
            after = scrape()
            res["cells"][f"{kind}-c{conc}"] = {"client": run.summary(),
                                               "server_phase_means": phase_means(before, after)}
        res["status"] = "ok"
        return res
    except Exception as exc:
        res["status"], res["error"] = "failed", repr(exc)
        return res
    finally:
        stop_server(proc)
        log.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model_path = os.environ["QWEN_MODEL_PATH"]
    report: dict = {"model_path": model_path}
    for name, extra in [("nocache", []), ("prefixcache", ["--enable-prefix-caching"])]:
        report[name] = run_server(name, extra, model_path, a.out)
        (a.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("DONE", name, report[name]["status"], flush=True)


if __name__ == "__main__":
    main()
