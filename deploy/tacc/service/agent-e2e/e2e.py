"""First end-to-end check: the unchanged agent harness (mock tools) driven by the real vLLM service.

Runs as a CPU job on a *different* node from the service, so it also proves:
  - the address file in USERDIR is shared between jobs,
  - the service node name resolves and port 8000 is reachable from another node,
  - the harness's real prompts (system prompt + tools + schema) get valid actions from the real model.

Tools, catalog and plan validator are the harness's own synthetic mock (agent.examples.mock_run);
nothing here is a real recommendation.
"""
import asyncio
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from agent import AgentDependencies, AgentRequest, AgentRuntime
from agent.examples.mock_run import MockPlanValidator, demo_tools
from inference.client import ModelClient
from inference.model_port import VllmModelPort

USERDIR = Path(os.environ["TACC_USERDIR"])
BASE = USERDIR / "tacc-agentforge-service"
ADDRESS = BASE / "service-url.txt"
OUT = BASE / f"e2e-{os.environ.get('SLURM_JOB_ID', 'local')}"
WAIT_S = 1500
CAP = {"currency": "USD", "purchase_cap": "100.00"}
SCENARIOS = [
    ("zh-budget-ok", "推荐预算内适合通勤的无线耳机。", CAP),
    ("en-budget-ok", "Recommend wireless earbuds for commuting within my budget.", CAP),
    ("zh-no-budget", "帮我挑一副通勤用的无线耳机。", {}),
    ("zh-tight-budget", "推荐通勤用的无线耳机。", {"currency": "USD", "purchase_cap": "10.00"}),
]
VARIANTS = {"think-default": None, "think-off": False}
REPEATS = 2


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def wait_for_service():
    t0 = time.monotonic()
    while time.monotonic() - t0 < WAIT_S:
        if ADDRESS.exists():
            address = ADDRESS.read_text(encoding="utf-8").strip()
            if address:
                break
        time.sleep(5)
    else:
        raise SystemExit(f"no service address after {WAIT_S}s: start the service job first")
    parts = urlsplit(address)
    info = {"address": address, "waited_s": round(time.monotonic() - t0, 1),
            "this_node": socket.gethostname(), "service_node": parts.hostname}
    info["same_node"] = info["this_node"] == info["service_node"]
    info["resolved_ip"] = socket.gethostbyname(parts.hostname)       # raises if the name does not resolve
    t1 = time.monotonic()
    with socket.create_connection((parts.hostname, parts.port or 80), timeout=10):
        info["tcp_connect_s"] = round(time.monotonic() - t1, 4)
    info["health_status"] = urlopen(f"http://{parts.hostname}:{parts.port}/health", timeout=10).status
    return info


async def one(client, name, message, constraints, variant, rep):
    deltas = {}                                   # operation_id -> counts and first-delta time

    def sink(op_id, kind, text):
        d = deltas.setdefault(op_id, {"content": 0, "reasoning": 0, "first_delta_s": None})
        d[kind] += 1
        if d["first_delta_s"] is None:
            d["first_delta_s"] = round(time.monotonic() - t0, 2)

    port = VllmModelPort(client, enable_thinking=VARIANTS[variant], on_delta=sink)
    runtime = AgentRuntime(AgentDependencies(port, demo_tools(), MockPlanValidator()))
    run_id = f"e2e-{name}-{variant}-{rep}"
    request = AgentRequest(run_id=run_id, conversation_id=f"c-{run_id}", snapshot_id="demo-snapshot-v1",
                           message=message, constraints=constraints)
    rec = {"scenario": name, "variant": variant, "rep": rep}
    t0 = time.monotonic()
    timeline = []
    try:
        result = None
        async for ev in runtime.astream(request):
            label = ev.type + (f":{ev.data['action_type']}" if "action_type" in ev.data else "") \
                + (f":{ev.data['code']}" if "code" in ev.data else "")
            timeline.append([round(time.monotonic() - t0, 2), label])
            if ev.result is not None:
                result = ev.result
        rec.update(status=result.status, stop_reason=result.stop_reason,
                   total_cost=result.verified_plan.total_cost if result.verified_plan else None,
                   question=result.clarification.question if result.clarification else None,
                   question_chars=len(result.clarification.question) if result.clarification else None,
                   issues=[i.code for i in result.issues], usage=result.usage.model_dump())
    except Exception as exc:                      # one bad run must not hide the others
        rec.update(status="exception", error=f"{type(exc).__name__}: {exc}")
    rec.update(wall_s=round(time.monotonic() - t0, 2), timeline=timeline, deltas=deltas)
    return rec


def summarise(rows):
    out = {}
    for r in rows:
        key = f"{r['scenario']}/{r['variant']}"
        out.setdefault(key, []).append(f"{r['status']}({r.get('stop_reason') or '-'}) {r['wall_s']}s")
    return out


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import langgraph, pydantic                      # noqa: E401
    env = {"python": platform.python_version(), "pydantic": pydantic.VERSION,
           "langgraph": getattr(langgraph, "__version__", "?"), "job": os.environ.get("SLURM_JOB_ID")}
    log(f"env {env}")
    net = wait_for_service()
    log(f"service reachable: {net}")
    client = ModelClient(address_file=ADDRESS, model="qwen3.8-27b")
    report = {"env": env, "network": net, "runs": []}

    for rep in range(1, REPEATS + 1):
        for variant in VARIANTS:
            for name, message, constraints in SCENARIOS:
                rec = await one(client, name, message, constraints, variant, rep)
                report["runs"].append(rec)
                log(f"{name:16s} {variant:13s} rep{rep}: {rec['status']} ({rec.get('stop_reason')}) "
                    f"{rec['wall_s']}s calls={rec.get('usage', {}).get('model_calls')}")

    log("4 harness runs at once (think-off)")
    t0 = time.monotonic()
    conc = await asyncio.gather(*[one(client, "zh-budget-ok", SCENARIOS[0][1], CAP, "think-off", 100 + i)
                                  for i in range(4)])
    report["concurrent4"] = {"wall_s": round(time.monotonic() - t0, 2), "runs": conc,
                             "statuses": [r["status"] for r in conc], "run_wall_s": [r["wall_s"] for r in conc]}
    log(f"concurrent4: {report['concurrent4']['statuses']} wall {report['concurrent4']['wall_s']}s")

    report["summary"] = summarise(report["runs"])
    (OUT / "e2e.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    log(f"RESULTS_DIR={OUT}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
