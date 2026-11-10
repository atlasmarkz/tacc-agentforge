"""Open-loop load test and fault injection against the vLLM service, from a CPU job on another node.

Load phases (open loop): requests arrive as a Poisson process at a fixed rate whether or not earlier ones have
finished, so queueing shows up as latency instead of being hidden by a closed loop. Each request reproduces a
real harness call: the real system prefix (harness prompt + knowledge tools + action schema) plus a unique
synthetic tool observation padded to a prompt length drawn from job 28561, and an output pinned to the
recorded completion length (`ignore_eos`). Thinking is off. The raw client makes one attempt per request,
so every failure is counted. Escalation stops once a phase is saturated.

Fault phase: steady traffic through the normal retrying client (what the harness uses); after `--kill-at`
seconds this job writes the service address into `fault-request` and the service (started in `validate`
mode) kills vLLM. Measured: what callers saw, how fast, whether anything hung past its time budget, and how
long until the address was withdrawn.

All results: <out>/load.json. Times in `events` are wall-clock, comparable with the service's events.jsonl.
"""
import argparse
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from client import ModelClient, ModelClientError

MODEL = "qwen3.8-27b"
TITLES = ["Wireless Earbuds with Noise Cancelling", "USB-C 65W GaN Charger, 3 Ports", "Commuter Laptop Backpack",
          "Compact Mechanical Keyboard", "Bluetooth 5.3 Sport Earbuds", "Fast Charger for Phone and Laptop"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)


def filler_item(rng):
    h = lambda: f"{rng.getrandbits(96):024x}"                                  # noqa: E731
    return json.dumps({
        "variant_id": f"variant_{h()}", "title": rng.choice(TITLES), "category": "earbuds", "type": "unknown",
        "identity_status": "provider_group_unverified",
        "selected_price": {"offer_id": f"offer_{h()}", "evidence_id": f"ev_{h()}", "currency": "USD",
                           "price_minor": rng.randint(500, 20000), "shipping_minor": rng.randint(0, 999),
                           "tax_status": "unknown", "price_age_seconds": rng.randint(1000, 300000),
                           "unknown_fields": ["destination", "quantity", "tax", "variant_identity"]}},
        separators=(",", ":"))


def make_messages(system, target_tokens, rng, base_tokens, chars_per_token):
    user = json.dumps({"schema_version": "1", "run_id": f"load-{rng.getrandbits(48):012x}",
                       "conversation_id": "load", "snapshot_id": "snapshot_5bcd00723fb0b7589cbe50c8",
                       "message": "Recommend wireless earbuds for my daily commute. My budget is 100 USD.",
                       "constraints": {"currency": "USD", "purchase_cap": "100.00"}}, ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    need_chars = (target_tokens - base_tokens) * chars_per_token
    if need_chars > 200:
        items, size = [], 0
        while size < need_chars:
            item = filler_item(rng)
            items.append(item)
            size += len(item) + 1
        observation = ('{"observation":{"status":"ok","tool":"search_products","data":{"candidates":['
                       + ",".join(items) + ']}}}')
        messages += [{"role": "assistant", "content": '{"type":"call_tool","tool_name":"search_products",'
                      '"arguments":{"filters":{"category":"earbuds"},"limit":10}}'},
                     {"role": "user", "content": observation}]
    return messages


def calibrate(client, system, rng):
    """Measure the real prompt tokens of the fixed prefix and the characters per token of the filler."""
    base = client.chat(make_messages(system, 0, rng, 0, 1), max_tokens=1, timeout_s=120, enable_thinking=False)
    padded = make_messages(system, 0, rng, -2000, 3.0)      # adds about 6000 characters of filler
    filler_chars = sum(len(m["content"]) for m in padded[2:])
    big = client.chat(padded, max_tokens=1, timeout_s=120, enable_thinking=False)
    cpt = filler_chars / max(1, big.prompt_tokens - base.prompt_tokens)
    if not 1.0 < cpt < 8.0:                                   # implausible (e.g. a fake server): fall back
        cpt = 3.0
    return {"base_tokens": base.prompt_tokens, "chars_per_token": round(cpt, 3)}


def one_request(client, messages, out_tokens, timeout, origin, recs, lock):
    rec = {"sent_s": round(time.monotonic() - origin, 3), "out_tokens": out_tokens}
    first = []
    t0 = time.monotonic()
    try:
        r = client.chat(messages, max_tokens=out_tokens, temperature=0, timeout_s=timeout, enable_thinking=False,
                        extra_body={"ignore_eos": True},
                        on_delta=lambda kind, text: first or first.append(time.monotonic()))
        rec.update(ok=True, latency_s=round(time.monotonic() - t0, 3),
                   ttft_s=round(first[0] - t0, 3) if first else None, prompt_tokens=r.prompt_tokens,
                   completion_tokens=r.completion_tokens, attempts=r.attempts)
    except ModelClientError as exc:
        rec.update(ok=False, error=type(exc).__name__, detail=str(exc)[:200],
                   latency_s=round(time.monotonic() - t0, 3), delivered=bool(first))
    rec["done_s"] = round(time.monotonic() - origin, 3)
    with lock:
        recs.append(rec)


def schedule(rate, duration, rng, launch):
    """Poisson arrivals for `duration` seconds; `launch()` must not block."""
    origin = time.monotonic()
    t, threads = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            return origin, threads
        delay = origin + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        threads.append(launch(origin))


def summarise(recs, offered_rate, duration, wall):
    ok = [r for r in recs if r["ok"]]
    errors = {}
    for r in recs:
        if not r["ok"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    lat = [r["latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    return {
        "offered_rate": offered_rate, "sent": len(recs), "ok": len(ok), "errors": errors,
        "error_rate": round(1 - len(ok) / len(recs), 3) if recs else None,
        "throughput_rps": round(len(ok) / wall, 3), "output_tok_s": round(sum(r["completion_tokens"] or 0 for r in ok) / wall, 1),
        "latency_s": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99), "max": pct(lat, 1)},
        "ttft_s": {"p50": pct(ttft, 0.5), "p95": pct(ttft, 0.95)},
        "prompt_tokens_mean": (round(statistics.mean(p), 0)
                               if (p := [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]) else None),
        "duration_s": duration, "wall_s": round(wall, 1),
    }


def load_phase(client, wl, cal, rate, duration, timeout, rng):
    recs, lock = [], threading.Lock()

    def launch(origin):
        target, out = rng.choice(wl["pairs"])
        msgs = make_messages(wl["system"], target, rng, cal["base_tokens"], cal["chars_per_token"])
        th = threading.Thread(target=one_request, daemon=True,
                              args=(client, msgs, max(16, min(512, out)), timeout, origin, recs, lock))
        th.start()
        return th

    origin, threads = schedule(rate, duration, rng, launch)
    for th in threads:
        th.join(timeout + 15)
    return summarise(recs, rate, duration, time.monotonic() - origin), recs


def fault_phase(address_file, address, wl, cal, rate, duration, kill_at, timeout, rng, events):
    client = ModelClient(address_file=address_file, model=MODEL)       # normal retrying client
    recs, lock = [], threading.Lock()
    trigger = address_file.parent / "fault-request"
    marks = {}
    stop = threading.Event()

    def watch_address():
        while not stop.wait(0.2):
            if "address_gone_s" not in marks and not address_file.exists():
                marks["address_gone_s"] = round(time.monotonic() - origin_box[0], 3)
                events.append({"t": round(time.time(), 3), "event": "address_gone_seen"})

    origin_box = [time.monotonic()]

    def launch(origin):
        origin_box[0] = origin
        if "trigger_s" not in marks and time.monotonic() - origin >= kill_at:
            trigger.write_text(address + "\n", encoding="utf-8")
            marks["trigger_s"] = round(time.monotonic() - origin, 3)
            events.append({"t": round(time.time(), 3), "event": "fault_requested"})
            log(f"fault requested at {marks['trigger_s']}s")
        target, out = rng.choice(wl["pairs"])
        msgs = make_messages(wl["system"], target, rng, cal["base_tokens"], cal["chars_per_token"])
        th = threading.Thread(target=one_request, daemon=True,
                              args=(client, msgs, max(16, min(512, out)), timeout, origin, recs, lock))
        th.start()
        return th

    threading.Thread(target=watch_address, daemon=True).start()
    origin, threads = schedule(rate, duration, rng, launch)
    for th in threads:
        th.join(timeout + 15)
    stop.set()
    t_kill = marks.get("trigger_s")
    after = [r for r in recs if t_kill is not None and r["done_s"] >= t_kill]
    first_error = min((r["done_s"] for r in after if not r["ok"]), default=None)
    outcomes = {}
    for r in after:
        key = "ok" if r["ok"] else r["error"]
        outcomes[key] = outcomes.get(key, 0) + 1
    return {
        "rate": rate, "duration_s": duration, "timeout_s": timeout, "sent": len(recs),
        "trigger_s": t_kill, "address_gone_s": marks.get("address_gone_s"),
        "address_gone_after_trigger_s": (round(marks["address_gone_s"] - t_kill, 3)
                                         if t_kill is not None and "address_gone_s" in marks else None),
        "first_error_after_trigger_s": round(first_error - t_kill, 3) if first_error is not None else None,
        "outcomes_after_trigger": outcomes,
        "in_flight_at_trigger": sum(1 for r in recs if t_kill is not None and r["sent_s"] < t_kill <= r["done_s"]),
        "hung_past_budget": sum(1 for r in recs if r["latency_s"] > timeout + 2),
        "max_latency_s": max((r["latency_s"] for r in recs), default=None),
    }, recs


def wait_for_address(address_file, wait_s):
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if address_file.exists() and address_file.read_text(encoding="utf-8").strip():
            return address_file.read_text(encoding="utf-8").strip(), round(time.monotonic() - t0, 1)
        time.sleep(5)
    raise SystemExit(f"no service address in {address_file} after {wait_s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--address-file", type=Path,
                    default=Path(os.environ.get("TACC_USERDIR", ".")) / "tacc-agentforge-service/service-url.txt")
    ap.add_argument("--workload", type=Path, default=Path(__file__).with_name("workload.json"))
    ap.add_argument("--rates", default="0.25,0.5,1,1.5,2,3")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--skip-fault", action="store_true")
    ap.add_argument("--fault-rate", type=float, default=0.5)
    ap.add_argument("--fault-duration", type=float, default=90)
    ap.add_argument("--kill-at", type=float, default=20)
    ap.add_argument("--fault-timeout", type=float, default=30)
    ap.add_argument("--wait-s", type=float, default=1500)
    ap.add_argument("--seed", type=int, default=6000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    wl = json.loads(a.workload.read_text(encoding="utf-8"))
    events = []
    report = {"job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename if hasattr(os, "uname") else None,
              "workload": {"source": wl["source"], "calls": len(wl["pairs"])}, "events": events}

    address, waited = wait_for_address(a.address_file, a.wait_s)
    parts = urlsplit(address)
    report["service"] = {"address": address, "waited_s": waited,
                         "health": urlopen(f"http://{parts.hostname}:{parts.port}/health", timeout=10).status}
    log(f"service {address} (waited {waited}s)")
    raw = ModelClient(base_url=address, model=MODEL, max_attempts=1)      # one attempt: count every failure
    report["calibration"] = cal = calibrate(raw, wl["system"], rng)
    log(f"calibration {cal}")

    report["phases"] = []
    for rate in [float(x) for x in a.rates.split(",")]:
        events.append({"t": round(time.time(), 3), "event": "phase_start", "rate": rate})
        summary, recs = load_phase(raw, wl, cal, rate, a.duration, a.timeout, rng)
        report["phases"].append({**summary, "requests": recs})
        log(f"rate {rate}/s: sent {summary['sent']} ok {summary['ok']} errors {summary['errors']} "
            f"p50 {summary['latency_s']['p50']}s p95 {summary['latency_s']['p95']}s "
            f"ttft p50 {summary['ttft_s']['p50']}s tput {summary['throughput_rps']}/s")
        if (summary["error_rate"] or 0) > 0.2 or (summary["latency_s"]["p95"] or 0) > a.timeout / 2:
            report["saturated_at_rate"] = rate
            log(f"saturated at {rate}/s; not escalating further")
            break

    if not a.skip_fault:
        events.append({"t": round(time.time(), 3), "event": "fault_phase_start"})
        summary, recs = fault_phase(a.address_file, address, wl, cal, a.fault_rate, a.fault_duration,
                                    a.kill_at, a.fault_timeout, rng, events)
        report["fault"] = {**summary, "requests": recs}
        log(f"fault: {summary['outcomes_after_trigger']} first error +{summary['first_error_after_trigger_s']}s "
            f"address gone +{summary['address_gone_after_trigger_s']}s hung {summary['hung_past_budget']}")

    (a.out / "load.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"RESULTS_DIR={a.out}")


if __name__ == "__main__":
    main()
