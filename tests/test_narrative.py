"""The streamed "why this pick" paragraph (app/narrative.py and the harness service's /runs/<id>/narrative)."""
import asyncio
import json
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pydantic", minversion="2")

ROOT = Path(__file__).resolve().parents[1]


def plan(items, total="23.99", left="76.01", breakdown=None):
    return SimpleNamespace(items=[SimpleNamespace(offer_id=o, quantity=q, evidence_ids=[]) for o, q in items], currency="USD",
                           total_cost=total, remaining_funds=left, cost_breakdown=breakdown or {"items": "23.99", "shipping": "0.00"})


LOOK = {
    "offers": [{"offer_id": "o1", "variant_id": "v1", "store_title": "Anker 65W 3-Port Charger, Foldable", "price_minor": 2399,
                "shipping_minor": 0, "shipping_status": "known"}],
    "evidence": [{"variant_id": "v1", "kind": "attribute", "text": "Output: 65W"},
                 {"variant_id": "v1", "kind": "offer", "text": "Seller: Walmart; item price: $23.99"}],
    "shops": {"o1": {"merchant": "Walmart"}}, "categories": {"v1": "chargers"},
}


def test_the_model_sees_only_the_verified_facts_and_the_shoppers_words():
    from app.narrative import build_request
    request, allowed = build_request(plan([("o1", 1)]), LOOK, "a charger for my trip to Sanya", {"purchase_cap": "100", "x": [1]}, "zh")
    system, user = request.messages[0].content, json.loads(request.messages[1].content)
    assert "Simplified Chinese" in system and request.action_schema is None
    item = user["items"][0]
    assert item["category"] == "chargers" and item["unit_price"] == "23.99" and item["shipping"] == "free"
    assert item["facts"] == ["Output: 65W"], "offer records are prices, not product facts"
    assert user["share_of_budget_percent"] == 24 and user["tax_included"] is False
    assert user["shopper_answers"] == {"purchase_cap": "100"}                       # only plain values are passed on
    assert {Decimal("65"), Decimal("23.99"), Decimal("100")} <= allowed


def test_a_sentence_with_an_invented_number_is_dropped_and_the_rest_flows():
    from app.narrative import SentenceGate, numbers_in
    gate = SentenceGate(numbers_in("price 23.99 total 65W budget 100"))
    out = []
    for piece in ["三亚天气", "炎热，这款65W充电器", "只要23.99美元。", "它还打了5折。", "总体很划算。"]:
        out += gate.feed(piece)
    out += gate.flush()
    assert "".join(out) == "三亚天气炎热，这款65W充电器只要23.99美元。总体很划算。"
    assert gate.dropped == 1 and gate.sent == 2


def test_a_decimal_point_is_not_the_end_of_a_sentence():
    from app.narrative import SentenceGate, numbers_in
    gate = SentenceGate(numbers_in("13.59"))
    assert gate.feed("It costs 13.") == []          # "13." may still become "13.59"
    out = gate.feed("59 in total. Good value") + gate.flush()
    assert "".join(out) == "It costs 13.59 in total. Good value" and gate.dropped == 0


def test_markdown_marks_are_removed_and_paragraph_breaks_are_kept():
    from app.narrative import SentenceGate
    gate = SentenceGate(set())
    out = gate.feed("**Great** pick.\n\nSecond paragraph.") + gate.flush()
    assert "".join(out) == "Great pick.\n\nSecond paragraph."


def test_the_service_streams_the_paragraph_for_a_finished_run():
    from agent import RunPolicy
    from app.backend import HarnessClient
    from app.harness_service import HarnessService, prepare_kb, serve
    from app.selftest import ScriptedPort

    async def scenario():
        userdir = Path(tempfile.mkdtemp(prefix="narrative-test-"))
        snapshot_id, _, kb, _ = prepare_kb(ROOT / "configs/knowledge.json", userdir, userdir)
        service = HarnessService(model=ScriptedPort(), kb=kb, snapshot_id=snapshot_id, policy=RunPolicy(max_context_chars=36000),
                                 token="t0ken", info={"job_id": "t"}, explain_default=False)
        stop, ready = asyncio.Event(), asyncio.get_running_loop().create_future()
        server = asyncio.create_task(serve(service, "127.0.0.1", 0, ready=ready.set_result, stop=stop))
        port = await ready
        client = HarnessClient(lambda: (f"http://127.0.0.1:{port}", "t0ken"))
        result = {}

        def blocking():
            events = list(client.stream("/runs", {"run_id": "n-1", "conversation_id": "c", "message": "wireless earbuds",
                                                   "constraints": {"purchase_cap": "100.00", "currency": "USD"}, "explain": False}))
            result["completed"] = any(e[0] == "agent" and (e[1].get("result") or {}).get("status") == "completed" for e in events)
            story = list(client.stream("/runs/n-1/narrative", {"goal": "earbuds for my commute", "lang": "en"}))
            result["names"] = [e[0] for e in story]
            result["text"] = "".join(e[1]["text"] for e in story if e[0] == "narrative")
            try:
                list(client.stream("/runs/nope/narrative", {"goal": "x"}))
            except Exception as exc:
                result["unknown"] = str(exc)

        try:
            await asyncio.to_thread(blocking)
        finally:
            stop.set()
            await server
            service.close()
            shutil.rmtree(userdir, ignore_errors=True)
        return result

    result = asyncio.run(scenario())
    assert result["completed"]
    assert result["names"][-2:] == ["narrative_done", "end"] and "narrative" in result["names"]
    assert "suits a daily commute" in result["text"]                  # the scripted model's sentence came through the gate
    assert "run_not_found" in result["unknown"] or "404" in result["unknown"]
