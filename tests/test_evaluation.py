"""Self-checks for the Task 6 enumerator (evaluations/catalog.py) and scorer (evaluations/score.py)."""
import json
import os
import sys
from pathlib import Path

import pytest

from evaluations.catalog import (
    CHECK_FAIL, CHECK_PASS, CHECK_UNKNOWN, Offer, check_requirement, ground_truth, judge_offer, offer_total,
)
from evaluations.score import aggregate, asks_for_budget, case_bootstrap, paired_difference, score_record


def attr(status="known", value_type="number", b=None, num=None, unit=None, text=None):
    return {"status": status, "value_type": value_type, "bool_value": b, "numeric_value": num, "unit": unit,
            "text_value": text}


def offer(oid, vid, price, shipping=0, currency="USD", ship_status="known", conflict=0):
    return Offer(oid, vid, f"title {oid}", "shop", currency, price, ship_status,
                 shipping if ship_status == "known" else None, conflict, f"ev_{oid}")


class FakeCatalog:
    def __init__(self, offers, attrs, evidence=()):
        self._offers, self._attrs, self._evidence = offers, attrs, set(evidence)

    def snapshot_id(self):
        return "snap-test"

    def offers(self, category):
        return [o for o in self._offers if o.variant_id.startswith(category)]

    def attributes(self, variant_id):
        return self._attrs.get(variant_id, {})

    def offer_ids(self):
        return {o.offer_id for o in self._offers}

    def offer_category(self, oid):
        o = next((o for o in self._offers if o.offer_id == oid), None)
        return o.variant_id.split("-")[0] if o else None

    def evidence_ids(self, ids):
        return set(ids) & self._evidence


CASE = {"case_id": "c1", "category": "earbuds", "constraints": {"currency": "USD", "purchase_cap": "50.00"},
        "budget_basis": "pre_tax", "requirements": [{"attribute": "anc", "op": "eq", "value": True}]}
CATALOG = FakeCatalog(
    offers=[offer("o-cheap", "earbuds-a", 3000, 500),        # 35.00, anc true   -> feasible
            offer("o-pricey", "earbuds-a", 6000, 0),          # 60.00             -> over budget
            offer("o-noanc", "earbuds-b", 1000, 0),           # anc false         -> violates
            offer("o-noship", "earbuds-a", 1000, ship_status="unknown"),   # cost unknown
            offer("o-unknownattr", "earbuds-c", 1000, 0),     # anc unknown       -> unverifiable
            offer("o-eur", "earbuds-a", 1000, 0, currency="EUR"),           # other currency
            offer("o-charger", "chargers-x", 1000, 0)],
    attrs={"earbuds-a": {"anc": attr(value_type="boolean", b=1)},
           "earbuds-b": {"anc": attr(value_type="boolean", b=0)},
           "earbuds-c": {"anc": attr(status="unknown", value_type="boolean")}},
    evidence={"ev-1"})


# ---- enumerator ---------------------------------------------------------------------------------------

def test_check_requirement_kinds():
    assert check_requirement({"anc": attr(value_type="boolean", b=1)}, {"attribute": "anc", "op": "eq", "value": True}) == CHECK_PASS
    assert check_requirement({"anc": attr(value_type="boolean", b=0)}, {"attribute": "anc", "op": "eq", "value": True}) == CHECK_FAIL
    assert check_requirement({}, {"attribute": "anc", "op": "eq", "value": True}) == CHECK_UNKNOWN
    p = {"power": attr(num=65.0, unit="W")}
    assert check_requirement(p, {"attribute": "power", "op": "gte", "value": 65, "unit": "W"}) == CHECK_PASS
    assert check_requirement(p, {"attribute": "power", "op": "gte", "value": 66, "unit": "W"}) == CHECK_FAIL
    assert check_requirement(p, {"attribute": "power", "op": "lte", "value": 100, "unit": "W"}) == CHECK_PASS
    assert check_requirement(p, {"attribute": "power", "op": "gte", "value": 1, "unit": "kW"}) == CHECK_UNKNOWN   # no unit conversion
    t = {"switch_type": attr(value_type="string", text="Mechanical, Kailh Red")}
    assert check_requirement(t, {"attribute": "switch_type", "op": "contains_any", "value": ["kailh"]}) == CHECK_PASS
    assert check_requirement(t, {"attribute": "switch_type", "op": "contains_none", "value": ["kailh"]}) == CHECK_FAIL
    w = {"connectivity": attr(value_type="string", text="wireless")}
    assert check_requirement(w, {"attribute": "connectivity", "op": "contains_any", "value": ["wired"]}) == CHECK_FAIL


def test_offer_total_is_exact_and_refuses_unknowns():
    assert str(offer_total(offer("x", "earbuds-a", 1999, 699), CASE)) == "26.98"
    assert str(offer_total(offer("x", "earbuds-a", 1999, 699), {**CASE, "quantity": 2})) == "46.97"   # shipping once
    assert offer_total(offer("x", "earbuds-a", 1999, ship_status="unknown"), CASE) is None
    assert offer_total(offer("x", "earbuds-a", 1999, 0, currency="EUR"), CASE) is None
    assert offer_total(offer("x", "earbuds-a", 1999, 0, conflict=1), CASE) is None
    with pytest.raises(ValueError):
        offer_total(offer("x", "earbuds-a", 1999, 0), {**CASE, "budget_basis": "with_tax"})


def test_budget_boundary_is_inclusive():
    exact = {**CASE, "constraints": {"currency": "USD", "purchase_cap": "35.00"}}
    assert judge_offer(offer("o", "earbuds-a", 3000, 500), CATALOG.attributes("earbuds-a"), exact).status == "feasible"
    cent_less = {**CASE, "constraints": {"currency": "USD", "purchase_cap": "34.99"}}
    assert judge_offer(offer("o", "earbuds-a", 3000, 500), CATALOG.attributes("earbuds-a"), cent_less).status == "over_budget"


def test_ground_truth_classifies_every_offer():
    gt = ground_truth(CASE, CATALOG)
    assert gt["expected"] == "plan" and [f["offer_id"] for f in gt["feasible"]] == ["o-cheap"]
    assert gt["counts"] == {"feasible": 1, "over_budget": 1, "violates": 1, "unverifiable": 3}
    assert ground_truth({**CASE, "constraints": {"currency": "USD", "purchase_cap": "10.00"}}, CATALOG)["expected"] == "no_feasible_plan"
    assert ground_truth({**CASE, "constraints": {"currency": "USD"}}, CATALOG)["expected"] == "clarify"


# ---- scorer -------------------------------------------------------------------------------------------

def rec(kind, rep=0, config="B3", **outcome):
    return {"config": config, "case_id": "c1", "rep": rep, "outcome": {"kind": kind, **outcome}, "wall_s": 10.0,
            "model_calls": 3, "prompt_tokens": 5000, "completion_tokens": 100}


def plan(oid, qty=1, ev=("ev-1",)):
    return {"items": [{"offer_id": oid, "quantity": qty, "evidence_ids": list(ev)}]}


@pytest.mark.parametrize("outcome,expected_reason,success", [
    (("plan", plan("o-cheap")), "ok", True),
    (("plan", plan("o-pricey")), "over_budget", False),
    (("plan", plan("o-noanc")), "violates_requirement", False),
    (("plan", plan("o-noship")), "unverifiable_claim", False),
    (("plan", plan("o-unknownattr")), "unverifiable_claim", False),
    (("plan", plan("o-made-up")), "unknown_offer_id", False),
    (("plan", plan("o-charger")), "wrong_category", False),
    (("plan", plan("o-cheap", qty=2)), "wrong_quantity", False),
    (("plan", plan("o-cheap", ev=("ev-made-up",))), "unknown_evidence_id", False),
    (("ask_user", {"question": "Budget?", "required_fields": ["purchase_cap"]}), "unnecessary_clarification", False),
    (("error", {"error": "timeout"}), "error:timeout", False),
])
def test_score_when_a_plan_is_expected(outcome, expected_reason, success):
    kind, body = outcome
    gt = ground_truth(CASE, CATALOG)
    s = score_record(rec(kind, **body), CASE, gt, CATALOG)
    assert (s["reason"], s["success"]) == (expected_reason, success)


def test_score_when_no_plan_is_feasible_or_input_is_missing():
    infeasible = {**CASE, "constraints": {"currency": "USD", "purchase_cap": "10.00"}}
    gt = ground_truth(infeasible, CATALOG)
    assert score_record(rec("ask_user", question="Nothing fits.", required_fields=["purchase_cap"]), infeasible, gt, CATALOG)["success"]
    s = score_record(rec("plan", **plan("o-cheap")), infeasible, gt, CATALOG)
    assert s["reason"] == "plan_when_infeasible" and s["over_budget"] is True
    missing = {**CASE, "constraints": {"currency": "USD"}}
    gt = ground_truth(missing, CATALOG)
    assert score_record(rec("ask_user", question="Budget?", required_fields=["budget"]), missing, gt, CATALOG)["success"]
    assert score_record(rec("ask_user", question="Colour?", required_fields=["colour"]), missing, gt, CATALOG)["reason"] == "clarification_misses_budget"
    assert score_record(rec("plan", **plan("o-cheap")), missing, gt, CATALOG)["reason"] == "plan_without_required_input"


def test_budget_field_detection():
    assert asks_for_budget(["purchase_cap"]) and asks_for_budget(["max_price"]) and asks_for_budget(["预算"])
    assert not asks_for_budget(["colour", "size"]) and not asks_for_budget([])


def test_aggregate_reports_denominators_and_bvr():
    gt = ground_truth(CASE, CATALOG)
    records = [rec("plan", rep=0, **plan("o-cheap")), rec("plan", rep=1, **plan("o-pricey")),
               rec("ask_user", rep=2, question="Budget?", required_fields=["purchase_cap"])]
    scores = [score_record(r, CASE, gt, CATALOG) for r in records]
    a = aggregate(scores, records)["B3"]
    assert a["tsr"] == {"n": 1, "m": 3, "rate": 0.333}
    assert a["bvr"] == {"n": 1, "m": 2, "rate": 0.5} and a["plans_delivered"] == 2
    assert a["amount_error_rate"] is None and a["wall_s"]["p50"] == 10.0
    assert a["outcome_reasons"] == {"ok": 1, "over_budget": 1, "unnecessary_clarification": 1}


def test_bvr_is_not_applicable_without_plans():
    gt = ground_truth(CASE, CATALOG)
    records = [rec("error", error="x")]
    a = aggregate([score_record(records[0], CASE, gt, CATALOG)], records)["B3"]
    assert a["bvr"] == {"n": 0, "m": 0, "rate": None}


def test_case_bootstrap_keeps_repeats_together_and_paired_difference():
    boot = case_bootstrap({"a": [1, 1, 1], "b": [0, 0, 0]}, n=500)
    assert boot["mean"] == 0.5 and boot["cases"] == 2 and 0 <= boot["ci95"][0] <= boot["ci95"][1] <= 1
    scores = ([{"config": "B0", "case_id": c, "success": False} for c in "abcd"]
              + [{"config": "B3", "case_id": c, "success": c in "ab"} for c in "abcd"])
    d = paired_difference(scores, "B0", "B3", n=500)
    assert d["diff"] == 0.5 and d["cases"] == 4


# ---- the real case file against the real seed snapshot ------------------------------------------------

def test_case_file_is_well_formed_and_split_correctly():
    cases = [json.loads(l) for l in Path("evaluations/cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == 42
    assert sum(c["split"] == "dev" for c in cases) == 10 and sum(c["split"] == "heldout" for c in cases) == 32
    assert {c["category"] for c in cases} == {"earbuds", "chargers", "keyboards", "backpacks"}
    for c in cases:
        assert c["budget_basis"] == "pre_tax" and c["constraints"]["currency"] == "USD"
        assert all(r["op"] in ("eq", "gte", "lte", "contains_any", "contains_none") for r in c["requirements"])


@pytest.mark.skipif(os.name == "nt" and not sys.flags.utf8_mode,
                    reason="knowledge/store.py reads snapshot files without an explicit encoding, so the seed fails "
                           "verification under a non-UTF-8 Windows locale (issue #5); run with PYTHONUTF8=1")
def test_ground_truth_on_the_seed_snapshot(tmp_path):
    from knowledge.bootstrap import install_snapshot
    from knowledge.settings import load_settings
    from evaluations.catalog import Catalog
    s = load_settings("configs/knowledge.json", data_root=tmp_path, snapshots_root=tmp_path / "snaps")
    install_snapshot(s)
    cat = Catalog(s.snapshots_root / s.snapshot_id / "catalog.sqlite")
    try:
        cases = {json.loads(l)["case_id"]: json.loads(l)
                 for l in Path("evaluations/cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        assert ground_truth(cases["dev-01"], cat)["expected"] == "plan"
        assert ground_truth(cases["dev-03"], cat)["expected"] == "no_feasible_plan"   # ANC under 15 USD
        assert ground_truth(cases["dev-05"], cat)["expected"] == "clarify"
        gt = ground_truth(cases["dev-06"], cat)                                       # mechanical, 100 USD
        assert gt["expected"] == "plan" and gt["cheapest_feasible_total"] == "85"
    finally:
        cat.close()


@pytest.mark.skipif(os.name == "nt" and not sys.flags.utf8_mode, reason="seed install needs UTF-8 mode on Windows (#5)")
def test_runner_end_to_end_against_a_fake_model(tmp_path):
    """Plumbing check: every configuration runs, records are written, and scoring/report are produced.
    The fake model always asks for the budget, so only the 'clarify' case can succeed."""
    import argparse
    import asyncio
    from evaluations import run
    from tests.fake_vllm import FakeVllm
    ask = {"choices": [{"message": {"content": json.dumps({"type": "ask_user", "question": "What is your budget?",
                                                           "required_fields": ["purchase_cap"]})},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    srv = FakeVllm([(200, ask, 0)])
    try:
        address = tmp_path / "service-url.txt"
        address.write_text(srv.url + "\n")
        a = argparse.Namespace(cases=Path("evaluations/cases.jsonl"), split="dev", only="dev-01,dev-05",
                               configs="B0,B1,B2,B3,B3h", repeats=1, config=Path("configs/knowledge.json"),
                               userdir=tmp_path, address_file=address, out=tmp_path / "out", unseal_heldout=False)
        asyncio.run(run.run_all(a))
    finally:
        srv.close()
    report = json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))
    assert set(report["summary"]) == {"B0", "B1", "B2", "B3", "B3h"}
    for cfg, r in report["summary"].items():
        assert r["tsr"] == {"n": 1, "m": 2, "rate": 0.5}, cfg       # dev-05 (clarify) passes, dev-01 (plan) fails
        assert r["outcome_reasons"] == {"ok": 1, "unnecessary_clarification": 1}
    records = [json.loads(l) for l in (tmp_path / "out/records.jsonl").read_text(encoding="utf-8").splitlines()]
    single = [r for r in records if r["config"] == "B1" and r["case_id"] == "dev-01"][0]
    assert single["candidates"] == 10 and single["evidence"] > 0
    first_body = srv.requests[0]
    assert first_body["response_format"]["type"] == "json_schema"


def test_runner_waits_for_the_service_and_gives_up_cleanly(tmp_path):
    import threading
    import time as _time
    from evaluations.run import wait_for_service
    from tests.fake_vllm import FakeVllm
    srv = FakeVllm([(200, {"status": "ok"}, 0)])
    try:
        address = tmp_path / "service-url.txt"
        threading.Timer(1.0, lambda: address.write_text(srv.url + chr(10))).start()   # service becomes ready later
        t0 = _time.monotonic()
        got = wait_for_service(address, wait_s=15)
        assert got["address"] == srv.url and 0.5 < _time.monotonic() - t0 < 12
    finally:
        srv.close()
    with pytest.raises(SystemExit, match="no healthy service"):
        wait_for_service(tmp_path / "never.txt", wait_s=0.2)


def test_heldout_split_is_sealed(monkeypatch, capsys):
    from evaluations import run
    monkeypatch.setattr(sys, "argv", ["run.py", "--split", "heldout", "--out", "x"])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert "sealed" in str(exc.value)


def test_recall_and_mrr_at_5():
    from evaluations.retrieval import recall_mrr, score_rankings
    assert recall_mrr({"a", "b"}, ["x", "a", "y", "z", "w", "b"]) == (0.5, 0.5)     # b is rank 6: outside top 5
    assert recall_mrr({"a"}, ["x", "y", "z"]) == (0.0, 0.0)
    assert recall_mrr(set(), ["a"]) is None                                          # not applicable, never 1
    labels = [{"case_id": "c1", "evidence_id": "a", "relevant": True},
              {"case_id": "c1", "evidence_id": "x", "relevant": False},
              {"case_id": "c2", "evidence_id": "q", "relevant": False},              # labelled, none relevant
              {"case_id": "c3", "evidence_id": "a", "relevant": None}]               # not labelled yet
    rankings = {("c1", "keyword"): ["a", "x"], ("c1", "filtered"): ["x", "y", "a"]}
    s = score_rankings(labels, rankings)
    assert s["keyword"] == {"recall_at_5": 1.0, "mrr_at_5": 1.0, "applicable_cases": 1, "labelled_cases": 2,
                            "not_applicable": 1}
    assert s["filtered"]["mrr_at_5"] == 0.333


def test_strip_hint_keeps_the_harness_formatting():
    from evaluations.run import strip_hint
    req = {"schema_version": "1", "message": "earbuds", "constraints": {"currency": "USD", "purchase_cap": "60.00",
                                                                       "available_categories": ["earbuds"]}}
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": json.dumps(req, ensure_ascii=False)},
            {"role": "user", "content": '{"observation": {"status": "ok"}}'}]
    out = strip_hint(msgs)
    without = {**req, "constraints": {"currency": "USD", "purchase_cap": "60.00"}}
    assert out[1]["content"] == json.dumps(without, ensure_ascii=False)       # what B3 would have sent
    assert out[0] == msgs[0] and out[2] == msgs[2] and "available_categories" in msgs[1]["content"]


@pytest.mark.skipif(os.name == "nt" and not sys.flags.utf8_mode, reason="seed install needs UTF-8 mode on Windows (#5)")
def test_trace_recording_for_sft(tmp_path):
    import argparse
    import asyncio
    from evaluations import run
    from tests.fake_vllm import FakeVllm
    ask = {"choices": [{"message": {"content": json.dumps({"type": "ask_user", "question": "What is your budget?",
                                                           "required_fields": ["purchase_cap"]})},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    srv = FakeVllm([(200, ask, 0)])
    try:
        address = tmp_path / "service-url.txt"
        address.write_text(srv.url + "\n")
        a = argparse.Namespace(cases=Path("evaluations/cases.jsonl"), split="dev", only="dev-05",
                               configs="B0,B3h", repeats=2, config=Path("configs/knowledge.json"), userdir=tmp_path,
                               address_file=address, out=tmp_path / "out", unseal_heldout=False, wait_s=10,
                               temperature=0.7, record_traces=True)
        asyncio.run(run.run_all(a))
    finally:
        srv.close()
    traces = [json.loads(l) for l in (tmp_path / "out/traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 2 and all(t["config"] == "B3h" and t["score_success"] for t in traces)   # dev-05: clarify
    call = traces[0]["calls"][0]
    assert call["accepted"] and json.loads(call["reply"])["type"] == "ask_user"
    assert [m["role"] for m in call["messages"]][:2] == ["system", "user"]
    assert "available_categories" not in json.dumps(call["messages"], ensure_ascii=False)
    assert any(abs(b.get("temperature", 0) - 0.7) < 1e-9 for b in srv.requests)                   # sampling on
    assert "trace_calls" not in (tmp_path / "out/records.jsonl").read_text(encoding="utf-8")
