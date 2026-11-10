"""End-to-end check with the real knowledge tools: unchanged harness + real KB snapshot + real model.

Runs as a CPU job on a node different from the vLLM service. What it measures:
  - does the real model call the four knowledge tools with valid arguments (filters, candidate_set_id, ids)?
  - what does it end up doing: ask_user, or propose a plan?
  - are the proposed offer and evidence IDs real (they exist in the snapshot) and were they seen in context?
  - latency, calls and tokens per run, with and without thinking, and with 4 runs at once.

There is no budget validator yet (Task 3). A recording stand-in returns status "unavailable" and keeps the
proposals, so a proposed plan ends the run as incomplete/validation_unavailable and is only *checked for
grounding* here. Nothing in this script is a purchase recommendation.

  python e2e_kb.py                         # TACC job: wait for the service, run everything
  python e2e_kb.py --selftest --config configs/knowledge.json   # plumbing check with a scripted model
"""
import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from agent import AgentDependencies, AgentRequest, AgentRuntime, Issue, ModelReply, RunPolicy, ValidationResult
from knowledge.agent_tools import build_knowledge_tools
from knowledge.bootstrap import install_snapshot
from knowledge.service import KnowledgeError, KnowledgeService
from knowledge.settings import load_settings
from knowledge.store import verify_snapshot

HERE = Path(__file__).resolve().parent

CAP = lambda amount: {"currency": "USD", "purchase_cap": amount}      # noqa: E731
SCENARIOS = [
    ("earbuds-100", "Recommend wireless earbuds for my daily commute. My budget is 100 USD.", CAP("100.00")),
    ("backpack-zh-80", "我想买一个通勤用的双肩包，预算80美元。", CAP("80.00")),
    ("charger-60", "I need a USB-C charger for my laptop and phone, under 60 USD.", CAP("60.00")),
    ("keyboard-no-budget", "Which mechanical keyboard would you suggest for programming?", {}),
    ("earbuds-5", "I want good wireless earbuds for 5 USD.", CAP("5.00")),
]
VARIANTS = {"think-off": (False, 2), "think-default": (None, 1)}     # name -> (enable_thinking, repeats)
CATEGORIES = ["earbuds", "chargers", "keyboards", "backpacks"]       # for --hint (test-only workaround)
TUNED_CONTEXT_CHARS = 36000                                          # ~12K tokens, inside the 16K model window
TUNED_THINKING_OUTPUT_TOKENS = 2048                                  # reasoning tokens count against max_output_tokens


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class RecordingPort:
    """Wraps the real model port; records every call (the reply text is the chosen action)."""

    def __init__(self, inner):
        self.inner, self.calls, self.last_text = inner, [], ""

    async def generate(self, request, context):
        t0 = time.monotonic()
        self.last_text = "\n".join(m.content for m in request.messages)
        try:
            reply = await self.inner.generate(request, context)
        except Exception as exc:
            self.calls.append({"op": context.operation_id, "s": round(time.monotonic() - t0, 2),
                               "error": f"{type(exc).__name__}: {exc}"})
            raise
        self.calls.append({"op": context.operation_id, "s": round(time.monotonic() - t0, 2),
                           "request_chars": len(self.last_text), "prompt_tokens": reply.prompt_tokens,
                           "completion_tokens": reply.completion_tokens, "finish": reply.finish_reason,
                           "reply": reply.content[:700]})
        return reply


class RecordingValidator:
    """Stand-in for the missing budget validator (Task 3): keeps the proposal, never approves it."""

    def __init__(self):
        self.proposals = []

    async def validate(self, request, context):
        self.proposals.append(request.plan.model_dump(mode="json"))
        return ValidationResult(
            status="unavailable", proposal_fingerprint=request.proposal_fingerprint,
            issues=[Issue(code="validator_not_implemented", message="No budget validator exists yet (Task 3).")])


def check_grounding(kb, snapshot_id, plan, context_text):
    """Are the IDs in a proposed plan real, and were they visible to the model?"""
    offer_ids = [i["offer_id"] for i in plan["items"]]
    evidence_ids = sorted({e for i in plan["items"] for e in i["evidence_ids"]})
    out = {"offer_ids": offer_ids, "evidence_ids": evidence_ids,
           "offers_seen_in_context": [o in context_text for o in offer_ids],
           "evidence_seen_in_context": [e in context_text for e in evidence_ids]}
    try:
        out["unknown_offers"] = kb.get_offer_facts(snapshot_id, offer_ids)["unknown_ids"]
        out["unknown_evidence"] = kb.resolve_evidence(snapshot_id, evidence_ids)["unknown_ids"]
    except KnowledgeError as exc:
        out["lookup_error"] = f"{exc.code}"
    out["grounded"] = (not out.get("unknown_offers") and not out.get("unknown_evidence")
                       and "lookup_error" not in out)
    return out


async def one(kb, tools, snapshot_id, make_port, name, message, constraints, variant, rep, policy=None):
    deltas = {"content": 0, "reasoning": 0}

    def sink(op_id, kind, text):
        deltas[kind] += 1

    port = RecordingPort(make_port(variant, sink))
    validator = RecordingValidator()
    runtime = AgentRuntime(AgentDependencies(port, tools, validator), policy)
    run_id = f"kb-{name}-{variant}-{rep}"
    request = AgentRequest(run_id=run_id, conversation_id=f"c-{run_id}", snapshot_id=snapshot_id,
                           message=message, constraints=constraints)
    rec = {"scenario": name, "variant": variant, "rep": rep}
    t0 = time.monotonic()
    timeline, tools_used = [], []
    try:
        result = None
        async for ev in runtime.astream(request):
            label = ev.type
            for key in ("action_type", "tool", "code", "status"):
                if key in ev.data:
                    label += f":{ev.data[key]}"
            timeline.append([round(time.monotonic() - t0, 2), label])
            if ev.type == "tool_started":
                tools_used.append(ev.data.get("tool"))
            if ev.result is not None:
                result = ev.result
        rec.update(status=result.status, stop_reason=result.stop_reason,
                   question=result.clarification.question if result.clarification else None,
                   required_fields=result.clarification.required_fields if result.clarification else None,
                   issues=[i.code for i in result.issues], usage=result.usage.model_dump())
    except Exception as exc:
        rec.update(status="exception", error=f"{type(exc).__name__}: {exc}")
    rec.update(wall_s=round(time.monotonic() - t0, 2), tools_used=tools_used, timeline=timeline,
               rejected=[l for _, l in timeline if l.startswith(("tool_rejected", "action_rejected"))],
               model_calls=port.calls, proposals=validator.proposals, deltas=deltas)
    rec["grounding"] = [check_grounding(kb, snapshot_id, p, port.last_text) for p in validator.proposals]
    return rec


def prepare_kb(config_path, userdir, scratch_parent):
    settings = load_settings(config_path, data_root=userdir / "tacc-agentforge-data",
                             snapshots_root=userdir / "tacc-agentforge-data/data/knowledge/snapshots")
    installed = install_snapshot(settings)
    scratch = Path(tempfile.mkdtemp(prefix="kb-e2e-", dir=scratch_parent))
    query_root = scratch / "snapshots"
    shutil.copytree(settings.snapshots_root / settings.snapshot_id, query_root / settings.snapshot_id)
    verification = verify_snapshot(query_root / settings.snapshot_id)
    if not verification["valid"]:
        raise SystemExit("query snapshot failed verification")
    kb = KnowledgeService(query_root, **settings.service)
    return settings, installed, verification, kb, scratch


class ScriptedPort:
    """Selftest only: search -> offer facts -> propose a plan, driven by the observations."""

    async def generate(self, request, context):
        obs = [json.loads(m.content)["observation"] for m in request.messages
               if m.role == "user" and m.content.startswith('{"observation":')]
        if not obs:
            action = {"type": "call_tool", "tool_name": "search_products",
                      "arguments": {"filters": {"category": "earbuds"}, "limit": 3}}
        elif obs[-1].get("tool") == "search_products":
            offer = next(c["selected_price"]["offer_id"] for c in obs[-1]["data"]["candidates"]
                         if (c.get("selected_price") or {}).get("offer_id"))
            action = {"type": "call_tool", "tool_name": "get_offer_facts", "arguments": {"offer_ids": [offer]}}
        else:
            offer = obs[-1]["data"]["offers"][0]
            action = {"type": "propose_plan", "plan": {"items": [{
                "offer_id": offer["offer_id"], "quantity": 1,
                "evidence_ids": [offer["evidence_id"]]}]}}
        return ModelReply(content=json.dumps(action))


async def run_selftest(config, out_dir):
    userdir = Path(tempfile.mkdtemp(prefix="kb-selftest-"))
    settings, installed, verification, kb, scratch = prepare_kb(config, userdir, userdir)
    tools = build_knowledge_tools(kb)
    try:
        rec = await one(kb, tools, settings.snapshot_id, lambda variant, sink: ScriptedPort(),
                        "selftest", "Recommend earbuds, budget 100 USD.", CAP("100.00"), "scripted", 1)
    finally:
        tools.get("search_products").executor.close()
    print(json.dumps({k: rec[k] for k in ("status", "stop_reason", "tools_used", "rejected", "grounding")},
                     indent=2, ensure_ascii=False))
    (out_dir / "selftest.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = rec["status"] == "incomplete" and rec["tools_used"] == ["search_products", "get_offer_facts"] \
        and rec["grounding"] and rec["grounding"][0]["grounded"]
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


async def run_real(config, userdir, out_dir, tuned, hint):
    from e2e import ADDRESS, wait_for_service
    from inference.client import ModelClient
    from inference.model_port import VllmModelPort
    import langgraph, pydantic                                 # noqa: E401
    env = {"python": platform.python_version(), "pydantic": pydantic.VERSION, "job": os.environ.get("SLURM_JOB_ID")}
    settings, installed, verification, kb, scratch = prepare_kb(config, userdir, os.environ.get("SLURM_TMPDIR"))
    log(f"snapshot {settings.snapshot_id}: {installed['status']}, valid={verification['valid']}")
    net = wait_for_service()
    log(f"service reachable: {net}")
    client = ModelClient(address_file=ADDRESS, model="qwen3.8-27b")

    def make_port(variant, sink):
        return VllmModelPort(client, enable_thinking=VARIANTS[variant][0], on_delta=sink)

    tools = build_knowledge_tools(kb)
    report = {"env": env, "network": net, "snapshot_id": settings.snapshot_id, "runs": [],
              "options": {"tuned_policy": tuned, "category_hint": hint}}

    def policy_for(variant):
        if not tuned:
            return RunPolicy()                                  # harness defaults
        out = TUNED_THINKING_OUTPUT_TOKENS if VARIANTS[variant][0] is None else 512
        return RunPolicy(max_context_chars=TUNED_CONTEXT_CHARS, max_output_tokens=out)

    def with_hint(constraints):
        return {**constraints, "available_categories": CATEGORIES} if hint else constraints
    try:
        for variant, (_, repeats) in VARIANTS.items():
            for rep in range(1, repeats + 1):
                for name, message, constraints in SCENARIOS:
                    rec = await one(kb, tools, settings.snapshot_id, make_port, name, message, with_hint(constraints),
                                    variant, rep, policy_for(variant))
                    report["runs"].append(rec)
                    log(f"{name:19s} {variant:13s} rep{rep}: {rec['status']} ({rec.get('stop_reason')}) {rec['wall_s']}s "
                        f"tools={rec['tools_used']} rejected={len(rec['rejected'])} "
                        f"grounded={[g['grounded'] for g in rec['grounding']]}")
        log("4 runs at once (think-off)")
        t0 = time.monotonic()
        conc = await asyncio.gather(*[one(kb, tools, settings.snapshot_id, make_port, *SCENARIOS[0][:1],
                                          SCENARIOS[0][1], with_hint(SCENARIOS[0][2]), "think-off", 100 + i,
                                          policy_for("think-off")) for i in range(4)])
        report["concurrent4"] = {"wall_s": round(time.monotonic() - t0, 2), "statuses": [r["status"] for r in conc],
                                 "run_wall_s": [r["wall_s"] for r in conc], "runs": conc}
        log(f"concurrent4: {report['concurrent4']['statuses']} wall {report['concurrent4']['wall_s']}s")
    finally:
        tools.get("search_products").executor.close()
        shutil.rmtree(scratch, ignore_errors=True)

    summary = {}
    for r in report["runs"]:
        key = f"{r['scenario']}/{r['variant']}"
        summary.setdefault(key, []).append(
            f"{r['status']}({r.get('stop_reason') or '-'}) {r['wall_s']}s tools={r['tools_used']}")
    report["summary"] = summary
    (out_dir / "e2e-kb.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    log(f"RESULTS_DIR={out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--config", type=Path, default=HERE / "pkg/configs/knowledge.json")
    parser.add_argument("--tuned", action="store_true", help="host policy: 36000 context chars, 2048 output tokens when thinking")
    parser.add_argument("--hint", action="store_true", help="test-only: put the valid category names into the constraints")
    args = parser.parse_args()
    if args.selftest:
        out = Path(tempfile.mkdtemp(prefix="kb-selftest-out-"))
        return asyncio.run(run_selftest(args.config, out))
    userdir = Path(os.environ["TACC_USERDIR"])
    out = userdir / "tacc-agentforge-service" / f"e2e-kb-{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run_real(args.config, userdir, out, args.tuned, args.hint))


if __name__ == "__main__":
    sys.exit(main())
