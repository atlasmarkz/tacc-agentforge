"""The follow-up question form (app/formspec.py and the harness service's /runs/<id>/form)."""
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pydantic", minversion="2")

ROOT = Path(__file__).resolve().parents[1]

GOOD = {"steps": [
    {"field": "item_type", "label": "要买的东西", "ask": "去巴厘岛，你想带回哪类礼物？", "kind": "multi", "options": ["咖啡豆", "木雕", "沙滩巾", "咖啡豆"]},
    {"field": "gift", "label": "是否送人", "ask": "是买来送人的吗？", "kind": "yesno", "options": ["x"]},
    {"field": "not_asked", "label": "忽略", "ask": "不应出现", "kind": "text", "options": []},
]}


def test_the_model_is_told_the_goal_and_the_catalog_and_nothing_else_is_asked():
    from app.formspec import build_request
    request = build_request("去巴厘岛旅游", "你想买什么？", ["item_type"], "zh", ["chargers", "earbuds"], {"purchase_cap": "100"})
    system, data = request.messages[0].content, json.loads(request.messages[1].content)
    assert "Simplified Chinese" in system and "chargers, earbuds" in system and request.action_schema is None
    assert data["shopper_request"] == "去巴厘岛旅游" and data["fields"] == ["item_type"] and data["already_known"] == {"purchase_cap": "100"}


def test_only_valid_steps_for_the_asked_fields_survive_and_in_the_asked_order():
    from app.formspec import parse_steps
    steps = parse_steps("```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```", ["gift", "item_type"])
    assert [s["field"] for s in steps] == ["gift", "item_type"]                    # asked order; "not_asked" is dropped
    gift, items = steps
    assert gift["kind"] == "yesno" and gift["options"] == []                       # yes/no has no options of its own
    assert items["options"] == ["咖啡豆", "木雕", "沙滩巾"]                         # duplicates removed


def test_malformed_or_thin_replies_give_nothing_or_plain_text_steps():
    from app.formspec import parse_steps
    assert parse_steps("sorry, I cannot", ["a"]) == [] and parse_steps(None, ["a"]) == [] and parse_steps('{"steps": 3}', ["a"]) == []
    thin = parse_steps(json.dumps({"steps": [{"field": "a", "label": "L", "ask": "Q?", "kind": "choice", "options": ["only one"]}]}), ["a"])
    assert thin == [{"field": "a", "label": "L", "ask": "Q?", "kind": "text", "options": []}]     # one option is not a choice
    assert parse_steps(json.dumps({"steps": [{"field": "a", "label": "", "ask": "Q?"}]}), ["a"]) == []


def test_the_service_words_the_question_of_a_waiting_run_only():
    from agent import ModelReply, RunPolicy
    from app.backend import HarnessClient, HarnessError
    from app.harness_service import HarnessService, prepare_kb, serve
    from app.selftest import ScriptedPort

    class Port(ScriptedPort):
        async def generate(self, request, context):
            if request.action_schema is None and "follow-up questions" in request.messages[0].content:
                return ModelReply(content=json.dumps(GOOD, ensure_ascii=False), finish_reason="stop")
            return await super().generate(request, context)

    async def scenario():
        userdir = Path(tempfile.mkdtemp(prefix="formspec-test-"))
        snapshot_id, _, kb, _ = prepare_kb(ROOT / "configs/knowledge.json", userdir, userdir)
        service = HarnessService(model=Port(), kb=kb, snapshot_id=snapshot_id, policy=RunPolicy(max_context_chars=36000),
                                 token="t0ken", info={"job_id": "t"}, explain_default=False)
        stop, ready = asyncio.Event(), asyncio.get_running_loop().create_future()
        server = asyncio.create_task(serve(service, "127.0.0.1", 0, ready=ready.set_result, stop=stop))
        port = await ready
        client = HarnessClient(lambda: (f"http://127.0.0.1:{port}", "t0ken"))
        result = {}

        def blocking():
            events = list(client.stream("/runs", {"run_id": "f-1", "conversation_id": "c", "message": "wireless earbuds",
                                                   "constraints": {}, "explain": False}))
            result["waiting"] = any(e[0] == "agent" and (e[1].get("result") or {}).get("status") == "waiting_user" for e in events)
            body = {"goal": "去巴厘岛旅游", "question": "你想买什么？", "fields": ["item_type", "gift"], "lang": "zh"}
            result["steps"] = client.post("/runs/f-1/form", body)["steps"]
            for name, path, payload in (("unknown", "/runs/nope/form", body), ("badfields", "/runs/f-1/form", {**body, "fields": ["x; drop"]}),
                                        ("nofields", "/runs/f-1/form", {**body, "fields": []})):
                try:
                    client.post(path, payload)
                except HarnessError as exc:
                    result[name] = exc.status

        try:
            await asyncio.to_thread(blocking)
        finally:
            stop.set()
            await server
            service.close()
            shutil.rmtree(userdir, ignore_errors=True)
        return result

    result = asyncio.run(scenario())
    assert result["waiting"]
    assert [s["field"] for s in result["steps"]] == ["item_type", "gift"] and result["steps"][0]["options"][0] == "咖啡豆"
    assert result["unknown"] == 404 and result["badfields"] == 400 and result["nofields"] == 400
