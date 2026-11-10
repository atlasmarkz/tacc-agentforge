import json

import pytest

from training.prepare_data import (
    DataError, build_examples, check_no_leakage, load_dev_cases, seen_categories, write_jsonl,
)

SEEN = {"earbuds", "chargers", "keyboards", "backpacks"}
SYSTEM = {"role": "system", "content": "You are a shopping advisor. Registered tools ..."}


def case(case_id, split="dev", category="earbuds", message="Recommend commuting earbuds under 100 USD"):
    return {"case_id": case_id, "split": split, "category": category, "message": message}


def call(reply, accepted=True, extra=()):
    return {"messages": [SYSTEM, {"role": "user", "content": '{"message":"earbuds"}'}, *extra],
            "reply": reply, "accepted": accepted}


SEARCH = '{"type": "call_tool", "tool_name": "search_products", "arguments": {"filters": {"category": "earbuds"}}}'
PLAN = '{"type":"propose_plan","plan":{"items":[{"offer_id":"offer_1","quantity":1,"evidence_ids":["ev_1"]}]}}'


def trace(case_id, calls, status="incomplete", stop_reason="validation_unavailable", grounded=True, run_id="r1"):
    return {"case_id": case_id, "run_id": run_id, "status": status, "stop_reason": stop_reason,
            "grounded": grounded, "calls": calls}


def write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_seen_categories_come_from_the_snapshot_schemas():
    assert seen_categories("configs/category_schemas.json") == SEEN


def test_load_dev_cases_refuses_heldout_and_unseen_category(tmp_path):
    assert len(load_dev_cases(write_cases(tmp_path, [case("d1"), case("d2", category="chargers")]), SEEN)) == 2
    with pytest.raises(DataError, match="split"):
        load_dev_cases(write_cases(tmp_path, [case("d1"), case("h1", split="heldout")]), SEEN)
    with pytest.raises(DataError, match="seen category"):
        load_dev_cases(write_cases(tmp_path, [case("d1", category="kettles")]), SEEN)
    with pytest.raises(DataError, match="duplicate"):
        load_dev_cases(write_cases(tmp_path, [case("d1"), case("d1")]), SEEN)


def test_build_examples_keeps_accepted_steps_of_good_runs_only():
    cases = [case("d1")]
    traces = [
        trace("d1", [call(SEARCH), call("{broken", accepted=False), call(PLAN)]),
        trace("d1", [call(SEARCH)], status="failed", stop_reason="invalid_arguments", run_id="r2"),
        trace("d1", [call(PLAN)], grounded=False, run_id="r3"),            # invented IDs: dropped
    ]
    examples = build_examples(cases, traces)
    assert [e.metadata["action_type"] for e in examples] == ["call_tool", "propose_plan"]
    first = examples[0]
    assert first.messages[-1]["role"] == "assistant"
    assert first.messages[-1]["content"] == json.dumps(json.loads(SEARCH), separators=(",", ":"))
    assert first.messages[:-1] == traces[0]["calls"][0]["messages"]


def test_clarification_runs_are_kept_and_duplicates_removed():
    ask = '{"type":"ask_user","question":"What is your budget?","required_fields":["purchase_cap"]}'
    examples = build_examples([case("d1")], [
        trace("d1", [call(ask)], status="waiting_user", stop_reason=None, grounded=None),
        trace("d1", [call(ask)], status="waiting_user", stop_reason=None, grounded=None, run_id="r2"),
    ])
    assert len(examples) == 1 and examples[0].metadata["action_type"] == "ask_user"


def test_trace_for_an_unknown_case_is_refused():
    with pytest.raises(DataError, match="not a loaded dev case"):
        build_examples([case("d1")], [trace("h1", [call(SEARCH)])])


def test_leakage_check_finds_heldout_text_anywhere():
    heldout = [case("h1", split="heldout", message="I need a USB-C charger for my laptop")]
    clean = build_examples([case("d1")], [trace("d1", [call(SEARCH)])])
    check_no_leakage(clean, heldout)
    leaky = build_examples([case("d1")], [trace("d1", [call(SEARCH, extra=[
        {"role": "user", "content": '{"observation":"... i need a  USB-C charger for my LAPTOP ..."}'}])])])
    with pytest.raises(DataError, match="held-out request text"):
        check_no_leakage(leaky, heldout)


def test_write_jsonl_is_deterministic_with_manifest(tmp_path):
    examples = build_examples([case("d1"), case("d2")], [
        trace("d2", [call(PLAN, extra=[{"role": "user", "content": "backpacks instead"}])], run_id="r9"),
        trace("d1", [call(SEARCH), call(PLAN)])])
    out = tmp_path / "train.jsonl"
    manifest = write_jsonl(examples, out)
    first = out.read_text(encoding="utf-8")
    write_jsonl(list(reversed(examples)), out)
    assert out.read_text(encoding="utf-8") == first
    rows = [json.loads(line) for line in first.splitlines()]
    assert [r["case_id"] for r in rows] == ["d1", "d1", "d2"]
    assert manifest["examples"] == 3 and manifest["cases"] == 2
    assert manifest["action_types"] == {"call_tool": 1, "propose_plan": 2}
    assert json.loads(out.with_suffix(".manifest.json").read_text())["sha256"] == manifest["sha256"]
