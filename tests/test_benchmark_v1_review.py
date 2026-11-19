"""Follow-ups to the PR #28 review: heldout cases must not repeat dev cases or each other, an adapter error ends
an episode without counting as an invalid action, and calls made before a runtime failure are still counted."""
from collections import Counter
import json
import re

import pytest

from evaluations.v1.oracle import truth
from evaluations.v1.suite import build


def _rows(tmp_path, fifth=False):
    out = tmp_path / ("fifth" if fifth else "base")
    build(out, fifth=fifth)
    rows = [json.loads(line) for line in (out / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return rows, json.loads((out / "catalogs.json").read_text(encoding="utf-8"))


def _content(case, catalog):
    """Everything a model or a tuner could see, with the case's own IDs masked."""
    case = {k: v for k, v in case.items() if k not in ("case_id", "split", "catalog_id")}
    text = json.dumps([case, catalog], sort_keys=True, ensure_ascii=False)
    return re.sub(r'"[^"]*(dev|heldout)-\d+[^"]*"', '"<id>"', text)


@pytest.mark.parametrize("fifth", [False, True])
def test_no_case_repeats_another(tmp_path, fifth):
    rows, catalogs = _rows(tmp_path, fifth)
    assert len(rows) == (50 if fifth else 42)
    # Neither the whole case nor the request alone may repeat: dialogue, ledger, basket and uncertainty requests
    # do not mention prices, so different catalogs alone are not enough (PR #31 review).
    assert max(Counter(_content(r, catalogs[r["catalog_id"]]) for r in rows).values()) == 1
    assert max(Counter(_content(r, None) for r in rows).values()) == 1


def test_committed_data_is_what_the_generator_builds(tmp_path):
    rows, catalogs = _rows(tmp_path)
    committed = [json.loads(line) for line in open("evaluations/v1/data/cases.jsonl", encoding="utf-8") if line.strip()]
    assert committed == rows
    assert json.load(open("evaluations/v1/data/catalogs.json", encoding="utf-8")) == catalogs


def test_dev_keeps_its_prices_and_heldout_keeps_its_answer_mix(tmp_path):
    rows, catalogs = _rows(tmp_path)
    dev = [r for r in rows if r["split"] == "dev"]
    # Dev is unchanged (base price 10), so the committed dev runs stay valid.
    assert all(min(float(o["price"]) for o in catalogs[r["catalog_id"]]["offers"]) == 10 for r in dev)
    expected = Counter(truth(r, catalogs[r["catalog_id"]])["expected"] for r in rows if r["split"] == "heldout")
    assert expected == {"plan": 22, "insufficient_evidence": 6, "no_feasible_plan": 3, "clarify": 1}


def test_an_adapter_error_ends_the_episode_without_an_invalid_action(tmp_path, monkeypatch):
    from evaluations.v1 import run
    rows, catalogs = _rows(tmp_path)
    case = next(r for r in rows if r["script"])                         # a scripted multi-turn case
    calls = []

    def respond(adapter, payload, timeout):
        calls.append(payload)
        return {"outcome": {"kind": "error"}}                           # no error or stop_reason field
    monkeypatch.setattr(run, "invoke", respond)
    rec = run.run_episode("unused:adapter", case, catalogs[case["catalog_id"]], "B3", 10)
    assert len(calls) == 1
    assert rec["stop_reason"] == "adapter_error" and not rec.get("invalid_actions")


def test_calls_before_a_runtime_failure_are_counted(monkeypatch):
    from evaluations.v1 import agent_adapter

    def extract_intent(client, payload, calls):
        calls.append({"prompt_tokens": 10, "completion_tokens": 5, "messages": []})
        return []

    async def run_agent(client, payload, requirements, calls=None):
        calls.append({"prompt_tokens": None, "completion_tokens": None, "messages": []})   # sent, never answered
        raise RuntimeError("model went away")
    monkeypatch.setattr(agent_adapter, "extract_intent", extract_intent)
    monkeypatch.setattr(agent_adapter, "run_agent", run_agent)
    rec = agent_adapter.respond({"config": "B3", "catalog": {"catalog_id": "x"}}, client=object())
    assert rec["outcome"] == {"kind": "error"} and rec["error"] == "RuntimeError"
    assert rec["model_calls"] == 2 and len(rec["model_trace"]) == 2
    assert rec["prompt_tokens"] is None                                  # an unanswered call has no token count
