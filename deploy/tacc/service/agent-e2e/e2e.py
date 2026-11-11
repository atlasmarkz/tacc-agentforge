"""End-to-end check: the agent harness (mock tools) driven by the real vLLM service.

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
from uuid import uuid4

from agent import AgentDependencies, AgentRequest, AgentRuntime, RunPolicy
from agent.checkpoint_cleanup import cleanup_checkpoints
from agent.checkpoints import open_checkpointer
from agent.examples.mock_run import MockEvidenceResolver, MockPlanValidator, demo_tools
from inference.client import ModelClient
from inference.model_port import VllmModelPort

USERDIR = Path(os.environ["TACC_USERDIR"])
BASE = USERDIR / "tacc-agentforge-service"
ADDRESS = BASE / "service-url.txt"
OUT = BASE / f"e2e-{os.environ.get('SLURM_JOB_ID', 'local')}"
CHECKPOINT_DB = Path(os.environ.get("AGENT_CHECKPOINT_DB", USERDIR / "tacc-agentforge/checkpoints/agent.sqlite"))
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


async def one(client, saver, session_id, name, message, constraints, variant, rep):
    deltas = {}                                   # operation_id -> counts and first-delta time

    def sink(op_id, kind, text):
        d = deltas.setdefault(op_id, {"content": 0, "reasoning": 0, "first_delta_s": None})
        d[kind] += 1
        if d["first_delta_s"] is None:
            d["first_delta_s"] = round(time.monotonic() - t0, 2)

    port = VllmModelPort(client, enable_thinking=VARIANTS[variant], on_delta=sink)
    runtime = AgentRuntime(AgentDependencies(port, demo_tools(), MockPlanValidator(),
                                              evidence_resolver=MockEvidenceResolver()),
                           RunPolicy(question_timeout_s=7200, explain=os.environ.get("AGENT_EXPLAIN", "0") == "1"), checkpointer=saver)
    run_id = f"e2e-{session_id}-{name}-{variant}-{rep}"
    request = AgentRequest(run_id=run_id, conversation_id=f"c-{run_id}", snapshot_id="demo-snapshot-v1",
                           message=message, constraints=constraints)
    rec = {"scenario": name, "variant": variant, "rep": rep, "run_id": run_id}
    t0 = time.monotonic()
    timeline = []
    explanation_start = explanation_first = explanation_done = None
    explanation_deltas = 0
    try:
        result = None
        async for ev in runtime.astream(request):
            timestamp = time.monotonic()
            if ev.type == "explain_started":
                explanation_start = timestamp
            elif ev.type == "explain_delta":
                explanation_first = explanation_first or timestamp
                explanation_deltas += 1
            elif ev.type == "explain_done":
                explanation_done = timestamp
            label = ev.type + (f":{ev.data['action_type']}" if "action_type" in ev.data else "") \
                + (f":{ev.data['code']}" if "code" in ev.data else "")
            timeline.append([round(time.monotonic() - t0, 2), label])
            if ev.result is not None:
                result = ev.result
        rec.update(status=result.status, stop_reason=result.stop_reason,
                   explanation=result.explanation.model_dump(mode="json") if result.explanation else None,
                   pending_question=result.pending_question.model_dump(mode="json") if result.pending_question else None,
                   total_cost=result.verified_plan.total_cost if result.verified_plan else None,
                   question=result.clarification.question if result.clarification else None,
                   question_chars=len(result.clarification.question) if result.clarification else None,
                   issues=[i.code for i in result.issues], usage=result.usage.model_dump())
        if result.pending_question is not None:
            # 即使后续场景/作业中断，日志中也已经有可提交答案的稳定身份。
            log("WAITING_RUN=" + json.dumps({"run_id": run_id, "pending_question": rec["pending_question"]}))
    except Exception as exc:                      # one bad run must not hide the others
        rec.update(status="exception", error=f"{type(exc).__name__}: {exc}")
    rec.update(wall_s=round(time.monotonic() - t0, 2), timeline=timeline, deltas=deltas)
    if explanation_start is not None:
        rec["explanation_metrics"] = {
            "first_delta_s": round(explanation_first - explanation_start, 4) if explanation_first else None,
            "duration_s": round(explanation_done - explanation_start, 4) if explanation_done else None,
            "content_delta_events": explanation_deltas,
            "chars": len(result.explanation.text) if result and result.explanation else 0,
        }
    return rec


def summarise(rows):
    out = {}
    for r in rows:
        key = f"{r['scenario']}/{r['variant']}"
        out.setdefault(key, []).append(f"{r['status']}({r.get('stop_reason') or '-'}) {r['wall_s']}s")
    return out


async def run_suite(saver):
    OUT.mkdir(parents=True, exist_ok=True)
    import langgraph, pydantic                      # noqa: E401
    env = {"python": platform.python_version(), "pydantic": pydantic.VERSION,
           "langgraph": getattr(langgraph, "__version__", "?"), "job": os.environ.get("SLURM_JOB_ID")}
    log(f"env {env}")
    net = wait_for_service()
    log(f"service reachable: {net}")
    client = ModelClient(address_file=ADDRESS, model="qwen3.8-27b")
    report = {"env": env, "network": net, "runs": []}
    session_id = uuid4().hex[:12]

    for rep in range(1, REPEATS + 1):
        for variant in VARIANTS:
            for name, message, constraints in SCENARIOS:
                rec = await one(client, saver, session_id, name, message, constraints, variant, rep)
                report["runs"].append(rec)
                log(f"{name:16s} {variant:13s} rep{rep}: {rec['status']} ({rec.get('stop_reason')}) "
                    f"{rec['wall_s']}s calls={rec.get('usage', {}).get('model_calls')}")

    log("4 harness runs at once (think-off)")
    t0 = time.monotonic()
    conc = await asyncio.gather(*[one(client, saver, session_id, "zh-budget-ok", SCENARIOS[0][1], CAP, "think-off", 100 + i)
                                  for i in range(4)])
    report["concurrent4"] = {"wall_s": round(time.monotonic() - t0, 2), "runs": conc,
                             "statuses": [r["status"] for r in conc], "run_wall_s": [r["wall_s"] for r in conc]}
    log(f"concurrent4: {report['concurrent4']['statuses']} wall {report['concurrent4']['wall_s']}s")

    report["summary"] = summarise(report["runs"])
    (OUT / "e2e.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    log(f"RESULTS_DIR={OUT}")


async def main():
    if os.environ.get("AGENT_SQLITE_STORAGE_VERIFIED") != "1":
        raise RuntimeError("Verify the checkpoint volume supports SQLite WAL, then set AGENT_SQLITE_STORAGE_VERIFIED=1.")
    async with open_checkpointer(CHECKPOINT_DB) as saver:
        for decision in await cleanup_checkpoints(saver):
            log(f"checkpoint cleanup: {decision.thread_id} {decision.action} {decision.reason}")
        await run_suite(saver)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
