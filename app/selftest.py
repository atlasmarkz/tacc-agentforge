"""Local check of the harness service and the backend without TACC: a scripted model drives the real harness,
the real knowledge tools (seed snapshot), the real budget validator and the real HTTP/SSE protocol, through the
same client the backend uses.

  python -m app.harness_service --selftest
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path

from agent import ModelReply, RunPolicy


class ScriptedPort:
    """No budget -> ask; else search -> facts of every candidate -> plan with the cheapest offer whose shipping
    is known; free text (the explanation) gets a short sentence."""

    async def generate(self, request, context):
        if request.action_schema is None:
            return ModelReply(content="This pair stays within your budget and suits a daily commute.", finish_reason="stop")
        user = next(m.content for m in request.messages if m.role == "user")
        obs = [json.loads(m.content)["observation"] for m in request.messages
               if m.role == "user" and m.content.startswith('{"observation":')]
        if "purchase_cap" not in user and not any("purchase_cap" in m.content for m in request.messages
                                                   if m.role == "user"):
            action = {"type": "ask_user", "question": "What is your budget in US dollars?", "required_fields": ["purchase_cap", "currency"]}
        elif not obs:
            action = {"type": "call_tool", "tool_name": "search_products",
                      "arguments": {"filters": {"category": "earbuds"}, "limit": 5}}
        elif obs[-1].get("tool") == "search_products":
            ids = [c["selected_price"]["offer_id"] for c in obs[-1]["data"]["candidates"]
                   if (c.get("selected_price") or {}).get("offer_id")]
            action = {"type": "call_tool", "tool_name": "get_offer_facts", "arguments": {"offer_ids": ids}}
        else:
            offers = [o for o in obs[-1]["data"]["offers"]
                      if o.get("shipping_status") == "known" and o.get("currency") == "USD"]
            offers.sort(key=lambda o: o["price_minor"] + o["shipping_minor"])
            offer = (offers or obs[-1]["data"]["offers"])[0]
            action = {"type": "propose_plan", "plan": {"items": [{
                "offer_id": offer["offer_id"], "quantity": 1, "evidence_ids": [offer["evidence_id"]]}]}}
        return ModelReply(content=json.dumps(action, ensure_ascii=False), finish_reason="stop")


async def run_selftest(config: Path) -> int:
    from app.harness_service import HarnessService, prepare_kb, serve
    from app.backend import HarnessClient
    userdir = Path(tempfile.mkdtemp(prefix="harness-selftest-"))
    snapshot_id, installed, kb, scratch = prepare_kb(config, userdir, userdir)
    service = HarnessService(model=ScriptedPort(), kb=kb, snapshot_id=snapshot_id,
                             policy=RunPolicy(max_context_chars=36000), token="t0ken",
                             info={"job_id": "selftest"}, explain_default=True)
    stop, ready = asyncio.Event(), asyncio.get_running_loop().create_future()
    server = asyncio.create_task(serve(service, "127.0.0.1", 0, ready=ready.set_result, stop=stop))
    port = await ready
    client = HarnessClient(lambda: (f"http://127.0.0.1:{port}", "t0ken"))
    checks = {}

    def blocking():
        checks["health"] = client.get("/health")["status"] == "ok"
        # 1. no budget: the run waits for an answer; 2. the answer resumes it to a verified plan
        events = list(client.stream("/runs", {"run_id": "st-1", "conversation_id": "c1",
                                               "message": "推荐通勤用的无线耳机", "constraints": {}}))
        waiting = [e for e in events if e[0] == "agent" and (e[1].get("result") or {}).get("status") == "waiting_user"]
        checks["asks_without_budget"] = bool(waiting)
        q = waiting[-1][1]["result"]["pending_question"]
        events = list(client.stream("/runs/st-1/resume", {"question_id": q["question_id"],
                                                          "answer": {"purchase_cap": "100.00", "currency": "USD"}}))
        final = [e[1]["result"] for e in events if e[0] == "agent" and e[1].get("result")]
        checks["resumed_to_plan"] = bool(final) and final[-1]["status"] == "completed" and bool(final[-1]["verified_plan"])
        checks["explanation"] = bool(final) and bool(final[-1].get("explanation"))
        checks["explain_streamed"] = any(e[0] == "agent" and e[1]["type"] == "explain_delta" for e in events)
        seqs = [e[1]["sequence"] for e in events if e[0] == "agent"]
        checks["sequence_increasing"] = seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        plan = final[-1]["verified_plan"] if final else None
        if plan:
            looked = client.post("/lookup", {"offer_ids": [i["offer_id"] for i in plan["items"]],
                                             "evidence_ids": [e for i in plan["items"] for e in i["evidence_ids"]]})
            checks["lookup_has_sources"] = bool(looked["evidence"]) and all(e.get("source_url") for e in looked["evidence"])
        # 3. a revision is a new run in the same conversation; 4. errors arrive as SSE error events
        events = list(client.stream("/runs", {"run_id": "st-2", "conversation_id": "c1", "message": "便宜一点的",
                                               "constraints": {"purchase_cap": "30.00", "currency": "USD"}}))
        checks["revision_finishes"] = any(e[0] == "end" for e in events)
        events = list(client.stream("/runs/st-1/resume", {"question_id": "x", "answer": {}}))
        checks["error_event"] = any(e[0] == "error" and e[1]["code"] == "not_waiting" for e in events)
        try:
            HarnessClient(lambda: (f"http://127.0.0.1:{port}", "wrong")).get("/health")
            checks["rejects_bad_token"] = False
        except Exception as exc:
            checks["rejects_bad_token"] = "401" in str(exc)

    try:
        await asyncio.to_thread(blocking)
    finally:
        stop.set()
        await server
        service.close()
        shutil.rmtree(userdir, ignore_errors=True)
    print(json.dumps(checks, indent=1))
    ok = all(checks.values())
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1
