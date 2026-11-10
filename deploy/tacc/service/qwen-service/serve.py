"""Run the vLLM API service inside one TACC job and keep it honest.

- Starts vLLM with the flags in service.json (public course environment, no source changes).
- "Ready" means a real generation succeeded (a constrained-JSON probe), not just /health.
- Only then is the service address published (atomic write to the address file). If the probe fails
  repeatedly, or vLLM exits, the address file is removed so clients see "unavailable" instead of hanging.
- Every probe interval appends one JSON line (latency, queue and request counters from /metrics) to probe.jsonl.
- With --smoke it also runs smoke.py against the live server; with --exit-after-smoke it then stops.
- GPU utilisation, memory and power are sampled every second into gpu.jsonl.
- vLLM exit is noticed within about a second (not only at the next probe), so the address is withdrawn fast.
- Test only: with --allow-fault-injection, a client can write this service's address into the file
  `fault-request` next to the address file, and vLLM is killed (SIGKILL). Timestamps go to events.jsonl.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from client import ModelClient, ModelClientError

PORT = 8000
PROBE_MESSAGES = [{"role": "user", "content": 'Reply with the JSON object {"ok": true} and nothing else.'}]
PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
METRICS = re.compile(r"^(vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc|"
                     r"request_success_total|num_preemptions_total|prompt_tokens_total|generation_tokens_total))"
                     r"(\{[^}]*\})? ([0-9.e+-]+)$")

stopping = False


def on_signal(signum, frame):
    global stopping
    stopping = True


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def vllm_command(cfg: dict, model_path: str) -> list[str]:
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", cfg["served_name"],
            "--host", "0.0.0.0", "--port", str(PORT),
            "--tensor-parallel-size", str(cfg["tensor_parallel_size"]), *cfg["args"]]


def probe(client: ModelClient, timeout_s: float = 30) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        r = client.chat(PROBE_MESSAGES, max_tokens=32, temperature=0, timeout_s=timeout_s,
                        json_schema=PROBE_SCHEMA, enable_thinking=False)
        ok = json.loads(r.content).get("ok") is True and r.finish_reason == "stop"
        return ok, time.monotonic() - t0, "" if ok else f"unexpected content {r.content[:80]!r}"
    except (ModelClientError, ValueError) as exc:
        return False, time.monotonic() - t0, f"{type(exc).__name__}: {exc}"


def scrape() -> dict:
    out: dict = {}
    try:
        text = urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5).read().decode()
    except OSError:
        return out
    for line in text.splitlines():
        m = METRICS.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(4))
    return out


def event(out: Path, name: str, **data) -> None:
    """Wall-clock timestamped events, comparable with the load job's own clock on another node."""
    with open(out / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "event": name, **data}) + "\n")


GPU_QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"]


def parse_gpu_csv(text: str) -> list[dict]:
    def num(value):
        try:
            return float(value)
        except ValueError:          # "[N/A]" or similar
            return None
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            gpus.append({"index": int(num(parts[0]) or 0), "util_pct": num(parts[1]), "mem_used_mib": num(parts[2]),
                         "mem_total_mib": num(parts[3]), "power_w": num(parts[4])})
    return gpus


def gpu_sampler(out: Path, done: threading.Event, interval: float = 1.0) -> None:
    with open(out / "gpu.jsonl", "a", encoding="utf-8") as f:
        while not done.wait(interval):
            try:
                text = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=5).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            f.write(json.dumps({"t": round(time.time(), 2), "gpus": parse_gpu_csv(text)}) + "\n")
            f.flush()


def fault_watcher(trigger: Path, address: str, proc: subprocess.Popen, out: Path, done: threading.Event) -> None:
    while not done.wait(0.5):
        try:
            target = trigger.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if target != address:           # never kill a service the request was not meant for
            continue
        trigger.unlink(missing_ok=True)
        event(out, "fault_kill", target=target)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError as exc:
            event(out, "fault_kill_failed", error=str(exc))
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--address-file", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("service.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exit-after-smoke", action="store_true")
    ap.add_argument("--probe-interval", type=float, default=30)
    ap.add_argument("--deadline-s", type=float, default=3400)
    ap.add_argument("--allow-fault-injection", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    model_path = os.environ["QWEN_MODEL_PATH"]
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    start = time.monotonic()
    host = socket.gethostname()
    address = f"http://{host}:{PORT}/v1"
    local = ModelClient(base_url=f"http://127.0.0.1:{PORT}/v1", model=cfg["served_name"], max_attempts=1)
    env = dict(os.environ, **cfg.get("env", {}))
    server_log = open(a.out / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(vllm_command(cfg, model_path), stdout=server_log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    info = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": host, "address": address, "config": cfg,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    published = False
    code = 0
    done = threading.Event()
    threading.Thread(target=gpu_sampler, args=(a.out, done), daemon=True).start()
    event(a.out, "vllm_started", pid=proc.pid)
    try:
        log(f"vLLM started pid={proc.pid}; waiting for a successful probe")
        while True:
            if stopping:
                return 143
            if proc.poll() is not None:
                log(f"vLLM exited during start-up, code {proc.returncode}")
                return 1
            if time.monotonic() - start > cfg["startup_timeout_s"]:
                log("start-up timeout")
                return 2
            try:
                urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            except OSError:
                time.sleep(5)
                continue
            ok, took, why = probe(local)
            log(f"probe ok={ok} {took:.2f}s {why}")
            if ok:
                break
            time.sleep(5)
        info["ready_after_s"] = round(time.monotonic() - start, 1)
        atomic_write(a.out / "service-info.json", json.dumps(info, indent=2))
        atomic_write(a.address_file, address + "\n")
        published = True
        log(f"READY and published {address} -> {a.address_file}")
        event(a.out, "ready", address=address, ready_after_s=info["ready_after_s"])
        if a.allow_fault_injection:
            trigger = a.address_file.parent / "fault-request"
            threading.Thread(target=fault_watcher, args=(trigger, address, proc, a.out, done), daemon=True).start()
            log(f"fault injection enabled: write {address} into {trigger}")

        if a.smoke:
            import smoke
            smoke.run(local, a.out, cfg)
            if a.exit_after_smoke:
                return 0

        failures = 0
        with open(a.out / "probe.jsonl", "a", encoding="utf-8") as probes:
            while not stopping and proc.poll() is None and time.monotonic() - start < a.deadline_s:
                # Wait for the next probe in short steps, so a vLLM exit is noticed within about a second.
                next_probe = time.monotonic() + a.probe_interval
                while time.monotonic() < next_probe and not stopping and proc.poll() is None:
                    time.sleep(0.5)
                if stopping or proc.poll() is not None:
                    break
                ok, took, why = probe(local)
                failures = 0 if ok else failures + 1
                probes.write(json.dumps({"t": round(time.monotonic() - start, 1), "ok": ok,
                                         "probe_s": round(took, 3), "error": why, "metrics": scrape()}) + "\n")
                probes.flush()
                if failures >= 3:
                    log("3 failed probes in a row; withdrawing the address")
                    a.address_file.unlink(missing_ok=True)
                    published = False
                    code = 3
                    break
                if not published and ok:
                    atomic_write(a.address_file, address + "\n")
                    published = True
        if proc.poll() is not None:
            log(f"vLLM exited, code {proc.returncode}")
            event(a.out, "vllm_exit_seen", code=proc.returncode)
            code = code or 1
        return code
    finally:
        done.set()
        try:
            if a.address_file.exists() and a.address_file.read_text(encoding="utf-8").strip() == address:
                a.address_file.unlink()          # never remove another job's newer address
                event(a.out, "address_withdrawn")
        except OSError:
            pass
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(60)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
