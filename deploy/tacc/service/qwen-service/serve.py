"""Run the vLLM API service inside one TACC job and keep it honest.

- Starts vLLM with the flags in service.json (public course environment, no source changes).
- "Ready" means a real generation succeeded (a constrained-JSON probe), not just /health.
- Only then is the service address published (atomic write to the address file). If the probe fails
  repeatedly, or vLLM exits, the address file is removed so clients see "unavailable" instead of hanging.
- Every probe interval appends one JSON line (latency, queue and request counters from /metrics) to probe.jsonl.
- With --smoke it also runs smoke.py against the live server; with --exit-after-smoke it then stops.
- GPU utilisation, memory and power are sampled every second into gpu.jsonl.
- vLLM exit is noticed within about a second (not only at the next probe), so the address is withdrawn fast.
- Recovery inside the job: after a vLLM crash or 3 failed probes, the address is withdrawn and vLLM is
  restarted (--max-restarts; only while --min-restart-time-s of job time remains), then re-published.
- Handover across the 1-hour job limit (--handover-lead-s > 0): that long before the deadline the job
  submits its successor (its own run.slurm) if service-chain.json allows; when the successor publishes its
  address, this job lets running requests finish and exits. Write service-stop to end the chain.
- Test only: with --allow-fault-injection, a client can write this service's address into the file
  `fault-request` next to the address file, and vLLM is killed (SIGKILL). Timestamps go to events.jsonl.
"""
import argparse
import json
import os
import re
import shutil
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


def nvml_reader():
    """GPU reader through NVML (vLLM bundles a copy), or None if NVML is unavailable."""
    try:
        try:
            import pynvml
        except ImportError:
            from vllm.third_party import pynvml
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return None

    def read():
        gpus = []
        for i, h in enumerate(handles):
            util, mem = pynvml.nvmlDeviceGetUtilizationRates(h), pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                power = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            except Exception:
                power = None
            gpus.append({"index": i, "util_pct": float(util.gpu), "mem_used_mib": round(mem.used / 2**20),
                         "mem_total_mib": round(mem.total / 2**20), "power_w": power})
        return gpus
    return read


def smi_reader():
    exe = shutil.which("nvidia-smi") or ("/usr/bin/nvidia-smi" if Path("/usr/bin/nvidia-smi").exists() else None)
    if exe is None:
        return None
    return lambda: parse_gpu_csv(subprocess.run([exe, *GPU_QUERY[1:]], capture_output=True, text=True,
                                                timeout=5, check=True).stdout)


def gpu_sampler(out: Path, done: threading.Event, interval: float = 1.0) -> None:
    """One line per second. The source used, or why sampling is impossible, goes to events.jsonl:
    round 1 (job 28565) recorded nothing and said nothing, because errors were swallowed."""
    read, source = nvml_reader(), "nvml"
    if read is None:
        read, source = smi_reader(), "nvidia-smi"
    if read is None:
        event(out, "gpu_sampler_unavailable", reason="neither NVML nor nvidia-smi is available")
        return
    event(out, "gpu_sampler_started", source=source)
    reported = False
    with open(out / "gpu.jsonl", "a", encoding="utf-8") as f:
        while not done.wait(interval):
            try:
                gpus = read()
            except Exception as exc:
                if not reported:                       # say it once, keep trying
                    event(out, "gpu_sampler_error", error=f"{type(exc).__name__}: {exc}"[:300])
                    reported = True
                continue
            f.write(json.dumps({"t": round(time.time(), 2), "gpus": gpus}) + "\n")
            f.flush()


def fault_watcher(trigger: Path, address: str, current: dict, out: Path, done: threading.Event) -> None:
    """Test only: kill the current vLLM each time a client writes this service's address into `trigger`."""
    while not done.wait(0.5):
        try:
            target = trigger.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if target != address:           # never kill a service the request was not meant for
            continue
        trigger.unlink(missing_ok=True)
        proc = current.get("proc")
        event(out, "fault_kill", target=target, pid=proc.pid if proc else None)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError) as exc:
            event(out, "fault_kill_failed", error=str(exc))


def stop_vllm(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(30)
        except Exception:
            pass


def wait_ready(proc, local, timeout_s) -> str:
    """'ready', 'exited', 'timeout' or 'stopping'."""
    t0 = time.monotonic()
    while True:
        if stopping:
            return "stopping"
        if proc.poll() is not None:
            return "exited"
        if time.monotonic() - t0 > timeout_s:
            return "timeout"
        try:
            urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        except OSError:
            time.sleep(5)
            continue
        ok, took, why = probe(local)
        log(f"probe ok={ok} {took:.2f}s {why}")
        if ok:
            return "ready"
        time.sleep(5)


def read_address(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def submit_successor(out: Path, chain_file: Path) -> None:
    """Submit the next service job (the same uploaded job directory) if the chain allows it."""
    try:
        chain = json.loads(chain_file.read_text(encoding="utf-8"))
        remaining = int(chain.get("remaining", 0))
    except (OSError, ValueError):
        remaining = 0
    if remaining <= 0 or (chain_file.parent / "service-stop").exists():
        event(out, "handover_skipped", remaining=remaining)
        return
    # serve.py runs inside the course runtime's proot root filesystem, which has no sbatch and no Slurm
    # configuration (jobs 28579 and 28581). job.sh runs a helper outside proot: we write a request file and it
    # submits the successor, then writes "ok <job id>" or "error <message>" to handover-result.
    request, result = out / "handover-request", out / "handover-result"
    result.unlink(missing_ok=True)
    atomic_write(request, json.dumps({"t": time.time()}))
    deadline = time.monotonic() + 90
    while not result.exists() and time.monotonic() < deadline:
        time.sleep(1)
    if not result.exists():
        event(out, "handover_submit_failed", error="no answer from the job.sh helper within 90 s")
        return
    status, _, detail = result.read_text(encoding="utf-8").strip().partition(" ")
    if status != "ok" or not detail.strip().isdigit():
        event(out, "handover_submit_failed", error=(detail or status)[:200])
        return
    successor = detail.strip()
    atomic_write(chain_file, json.dumps({"remaining": remaining - 1, "last_submitted": successor,
                                         "by": os.environ.get("SLURM_JOB_ID")}))
    event(out, "handover_submitted", successor=successor, remaining=remaining - 1)


def drain(out: Path, timeout_s: float = 120) -> None:
    """After a newer service took over: let the requests still running here finish (or time out)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s and not stopping:
        m = scrape()
        if m.get("vllm:num_requests_running", 0) + m.get("vllm:num_requests_waiting", 0) <= 0:
            break
        time.sleep(1)
    event(out, "drained", after_s=round(time.monotonic() - t0, 1))


def monitor(a, proc, local, start, address, handover_done: bool) -> str:
    """Watch a ready vLLM. Returns 'deadline', 'stopping', 'vllm_exit', 'unhealthy', 'superseded' or
    'handover_due'. Checks every 0.5 s; probes every --probe-interval seconds."""
    failures = 0
    with open(a.out / "probe.jsonl", "a", encoding="utf-8") as probes:
        while True:
            next_probe = time.monotonic() + a.probe_interval
            while time.monotonic() < next_probe:
                if stopping:
                    return "stopping"
                if proc.poll() is not None:
                    return "vllm_exit"
                elapsed = time.monotonic() - start
                if elapsed >= a.deadline_s:
                    return "deadline"
                if not handover_done and a.handover_lead_s > 0 and elapsed >= a.deadline_s - a.handover_lead_s:
                    return "handover_due"
                current = read_address(a.address_file)
                if current and current != address:
                    return "superseded"
                time.sleep(0.5)
            ok, took, why = probe(local)
            failures = 0 if ok else failures + 1
            probes.write(json.dumps({"t": round(time.monotonic() - start, 1), "ok": ok, "probe_s": round(took, 3),
                                     "error": why, "metrics": scrape()}) + "\n")
            probes.flush()
            if failures >= 3:
                return "unhealthy"
            if ok and not read_address(a.address_file):
                atomic_write(a.address_file, address + "\n")   # re-publish after a transient withdrawal


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
    ap.add_argument("--max-restarts", type=int, default=2,
                    help="restart vLLM inside this job after a crash or 3 failed probes, at most this often")
    ap.add_argument("--min-restart-time-s", type=float, default=600,
                    help="do not restart when less job time than this remains (start-up takes about 5 min)")
    ap.add_argument("--handover-lead-s", type=float, default=0,
                    help="if > 0: this many seconds before the deadline, submit the successor job "
                         "(allowed by service-chain.json next to the address file); 0 disables handover")
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
    info = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": host, "address": address, "config": cfg,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    chain_file = a.address_file.parent / "service-chain.json"
    done = threading.Event()
    current: dict = {}
    threading.Thread(target=gpu_sampler, args=(a.out, done), daemon=True).start()
    if a.allow_fault_injection:
        trigger = a.address_file.parent / "fault-request"
        threading.Thread(target=fault_watcher, args=(trigger, address, current, a.out, done), daemon=True).start()
        log(f"fault injection enabled: write {address} into {trigger}")
    restarts, handover_done, code = 0, False, 0

    def withdraw(reason: str) -> None:
        if read_address(a.address_file) == address:            # never remove another job's newer address
            a.address_file.unlink(missing_ok=True)
            event(a.out, "address_withdrawn", reason=reason)

    try:
        while True:
            attempt_start = time.monotonic()
            proc = subprocess.Popen(vllm_command(cfg, model_path), stdout=server_log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            current["proc"] = proc
            event(a.out, "vllm_started", pid=proc.pid, attempt=restarts)
            log(f"vLLM started pid={proc.pid} (attempt {restarts}); waiting for a successful probe")
            state = wait_ready(proc, local, cfg["startup_timeout_s"])
            if state == "stopping":
                return 143
            if state != "ready":
                reason = f"startup_{state}"
            else:
                info["ready_after_s"] = round(time.monotonic() - attempt_start, 1)
                info["restarts"] = restarts
                atomic_write(a.out / "service-info.json", json.dumps(info, indent=2))
                atomic_write(a.address_file, address + "\n")
                log(f"READY and published {address} -> {a.address_file}")
                event(a.out, "ready", address=address, ready_after_s=info["ready_after_s"], attempt=restarts)
                if a.smoke:
                    import smoke
                    smoke.run(local, a.out, cfg)
                    if a.exit_after_smoke:
                        return 0
                reason = monitor(a, proc, local, start, address, handover_done)
                if reason == "handover_due":
                    submit_successor(a.out, chain_file)
                    handover_done = True
                    reason = monitor(a, proc, local, start, address, handover_done)
            if reason == "stopping":
                return 143
            if reason == "deadline":
                return code
            if reason == "superseded":
                event(a.out, "superseded", by=read_address(a.address_file))
                drain(a.out)
                return 0
            # vLLM died or stopped answering: withdraw, then restart if allowed and worthwhile.
            withdraw(reason)
            event(a.out, "vllm_down", reason=reason, code=proc.poll())
            stop_vllm(proc)
            left = a.deadline_s - (time.monotonic() - start)
            if restarts >= a.max_restarts or left < a.min_restart_time_s:
                event(a.out, "giving_up", reason=reason, restarts=restarts, time_left_s=round(left))
                return 1
            restarts += 1
            event(a.out, "restarting", reason=reason, attempt=restarts, time_left_s=round(left))
            time.sleep(5)                                      # let the GPUs and the port be released
    finally:
        done.set()
        withdraw("exit")
        if current.get("proc") is not None:
            stop_vllm(current["proc"])
        server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
