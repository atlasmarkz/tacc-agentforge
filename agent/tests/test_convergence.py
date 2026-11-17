"""预算收敛、真实冻结快照及跨恢复证据的行为回归，不访问模型服务。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

from langgraph.checkpoint.memory import InMemorySaver
import pytest
from pydantic import ValidationError

from agent import AgentDependencies, AgentRequest, AgentRuntime, CapabilityError, ModelReply, RunPolicy, ToolRegistry
from agent.budget import BudgetPlanValidator
from agent.contracts import AskUser, RunStateError
from agent.examples.mock_run import MockPlanValidator, MockSearch, demo_request, demo_tools, DEMO_OFFER, DEMO_EVIDENCE
from agent.state import initial_state

SEARCH = {"type": "call_tool", "tool_name": "search_products", "arguments": {"query": "earbuds"}}
PLAN = {"type": "propose_plan", "plan": {"items": [
    {"offer_id": DEMO_OFFER, "quantity": 1, "evidence_ids": [DEMO_EVIDENCE]},
]}}


class Scripted:
    def __init__(self, *actions):
        self.actions, self.requests = iter(actions), []

    async def generate(self, request, context):
        self.requests.append(request)
        return ModelReply(content=json.dumps(next(self.actions)))


class CountedSearch(MockSearch):
    def __init__(self, fail_first=False):
        self.calls, self.fail_first = 0, fail_first

    async def execute(self, arguments, context):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise CapabilityError("temporary_error", "Try again.", recoverable=True)
        return await super().execute(arguments, context)


def test_two_final_decisions_are_reserved_and_the_validator_still_runs():
    model = Scripted(*[deepcopy(SEARCH) for _ in range(6)], PLAN)
    runtime = AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator()),
                           RunPolicy(max_no_progress_calls=20))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.status == "completed" and result.usage.model_calls == 7
    assert result.usage.tool_calls == 6 and result.verified_plan.total_cost == "49.99"
    schema = model.requests[-1].action_schema
    assert "call_tool" not in schema["discriminator"]["mapping"]
    assert "ask_user" not in schema["discriminator"]["mapping"]
    assert '"model_calls_remaining": 2' in model.requests[-1].messages[0].content


def test_repeated_read_queries_reuse_success_and_force_a_proposal():
    async def scenario():
        search = CountedSearch()
        tool = replace(demo_tools(search).get("search_products"), read_only=True)
        model = Scripted(SEARCH, SEARCH, SEARCH, PLAN)
        runtime = AgentRuntime(AgentDependencies(model, ToolRegistry([tool]), MockPlanValidator()))
        result = await runtime.run(demo_request())
        assert result.status == "completed" and result.usage.model_calls == 4
        assert search.calls == result.usage.tool_calls == 1
        assert "call_tool" not in model.requests[-1].action_schema["discriminator"]["mapping"]
        trace = await runtime.export_trace(result.run_id)
        assert len([e for e in trace["entries"] if e.get("cached")]) == 2
        assert trace["progress"]["no_progress_calls"] == 2
        cached = json.loads(model.requests[-1].messages[-1].content)["observation"]
        assert cached["no_new_information"] is True
        assert cached["previous_operation_id"].endswith(":tool:1")
    asyncio.run(scenario())


def test_recoverable_error_is_not_cached_and_success_restores_progress():
    search = CountedSearch(fail_first=True)
    tool = replace(demo_tools(search).get("search_products"), read_only=True)
    runtime = AgentRuntime(AgentDependencies(Scripted(SEARCH, SEARCH, PLAN), ToolRegistry([tool]), MockPlanValidator()))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.status == "completed" and search.calls == result.usage.tool_calls == 2


@pytest.mark.parametrize("reason", ["evidence_insufficient", "requirements_unverified", "no_feasible_plan"])
def test_final_decision_can_abstain_without_inventing_a_plan(reason):
    message = "No cited evidence establishes the required simultaneous charging wattage."
    model = Scripted({"type": "report_incomplete", "reason": reason, "message": message})
    runtime = AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator()), RunPolicy(max_model_calls=1))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.status == "incomplete" and result.stop_reason == reason
    assert result.verified_plan is None and result.issues[-1].message == message
    assert result.message == message, "the user-facing page and the evaluator need the model's own wording"
    assert "report_incomplete" in model.requests[0].action_schema["discriminator"]["mapping"]


def test_search_page_without_a_known_satisfying_candidate_is_not_progress():
    """价格与硬约束都未知的搜索页（冻结基线 dev-07 的翻页）不构成进展。"""
    from agent.progress import record_result
    state = initial_state(demo_request(), RunPolicy())
    page = {"next_cursor": "c1", "candidates": [
        {"offer_id": "o-1", "selected_price": None, "constraint_status": "unknown",
         "hard_constraint_status": {"battery_life": "unknown"}}]}
    state["progress"] = record_result(state, "search_products", page)
    assert state["progress"]["no_progress_calls"] == 1
    state["progress"] = record_result(state, "search_products", {**page, "next_cursor": "c2"})
    assert state["progress"]["no_progress_calls"] == 2
    priced = {"candidates": [{"offer_id": "o-2", "selected_price": {"price_minor": 1999},
                              "constraint_status": "satisfied",
                              "hard_constraint_status": {"battery_life": "satisfied"}}]}
    state["progress"] = record_result(state, "search_products", priced)
    assert state["progress"]["no_progress_calls"] == 0


def test_correcting_an_error_leaves_one_targeted_evidence_read_before_stalling():
    from agent.progress import record_result
    state = initial_state(demo_request(), RunPolicy())
    state["progress"] = record_result(state, "search_products", {"code": "invalid_arguments"}, error=True)
    unknown = {"candidates": [{"variant_id": "v1", "selected_price": None,
                              "constraint_status": "unknown"}]}
    state["progress"] = record_result(state, "search_products", unknown)
    assert state["progress"]["no_progress_calls"] == 1
    state["progress"] = record_result(state, "search_products", unknown)
    assert state["progress"]["no_progress_calls"] == 2


def test_pagination_is_progress_and_response_timestamps_are_not():
    from agent.progress import record_result
    state = initial_state(demo_request(), RunPolicy())
    page1 = {"responded_at": "first", "evidence": [{"text": "45W top port", "text_start": 0}]}
    state["progress"] = record_result(state, "resolve_evidence", page1)
    state["progress"] = record_result(state, "resolve_evidence", {**page1, "responded_at": "later"})
    assert state["progress"]["no_progress_calls"] == 1
    page2 = {"responded_at": "later", "evidence": [{"text": "20W bottom port", "text_start": 1400}]}
    state["progress"] = record_result(state, "resolve_evidence", page2)
    assert state["progress"]["no_progress_calls"] == 0


def test_handles_cursors_order_and_ranking_do_not_manufacture_new_evidence():
    from agent.progress import record_result
    state = initial_state(demo_request(), RunPolicy())
    facts = [{"evidence_id": "e1", "text": "45W top port", "text_start": 0},
             {"evidence_id": "e2", "text": "20W bottom port", "text_start": 0}]
    state["progress"] = record_result(state, "retrieve_evidence", {
        "candidate_set_id": "first", "next_cursor": "page2", "evidence": facts})
    state["progress"] = record_result(state, "retrieve_evidence", {
        "candidate_set_id": "new-handle", "next_cursor": "page3",
        "evidence": [{**f, "rank": n, "score": n / 10} for n, f in enumerate(reversed(facts))]})
    assert state["progress"]["no_progress_calls"] == 1
    # 同一事实跨 retrieve/resolve 读取也不是进展；后续新文本片段仍是进展。
    state["progress"] = record_result(state, "resolve_evidence", {"evidence": facts[:1]})
    assert state["progress"]["no_progress_calls"] == 2
    state["progress"] = record_result(state, "resolve_evidence", {
        "evidence": [{**facts[0], "text": "USB PD supported", "text_start": 100}]})
    assert state["progress"]["no_progress_calls"] == 0


def test_different_empty_searches_are_discoveries_but_pagination_is_not():
    from agent.progress import record_result
    state = initial_state(demo_request(), RunPolicy())
    empty = {"candidates": [], "total_count": 0, "scope_complete": True, "next_cursor": None}
    for cap in (1000, 1500):
        state["progress"] = record_result(state, "search_products", empty,
                                           arguments={"filters": {"max_price_minor": cap}})
        assert state["progress"]["no_progress_calls"] == 0
    state["progress"] = record_result(state, "search_products", empty,
        arguments={"filters": {"max_price_minor": 1500}, "limit": 20, "cursor": "different"})
    assert state["progress"]["no_progress_calls"] == 1
    empty_evidence = {"candidate_set_id": "a", "evidence": []}
    state["progress"] = record_result(state, "retrieve_evidence", empty_evidence,
        arguments={"candidate_set_id": "a", "query": "charging"})
    state["progress"] = record_result(state, "retrieve_evidence", {**empty_evidence, "candidate_set_id": "b"},
        arguments={"candidate_set_id": "b", "query": "charging"})
    assert state["progress"]["no_progress_calls"] == 1


def test_last_decision_never_executes_a_tool_even_if_the_model_ignores_schema():
    search = CountedSearch()
    model = Scripted(*[deepcopy(SEARCH) for _ in range(8)])
    runtime = AgentRuntime(AgentDependencies(model, demo_tools(search)),
                           RunPolicy(finalize_reserve_model_calls=1, max_no_progress_calls=20))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.usage.model_calls == 8 and result.usage.tool_calls == search.calls == 7
    assert result.stop_reason == "convergence_failed"
    mapping = model.requests[-1].action_schema["discriminator"]["mapping"]
    assert set(mapping) == {"propose_plan", "report_incomplete"}


def test_eighth_decision_can_repair_a_proposal_without_raising_the_limit():
    model = Scripted(*[deepcopy(SEARCH) for _ in range(6)], {"type": "propose_plan", "plan": {}}, PLAN)
    runtime = AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator()),
                           RunPolicy(max_no_progress_calls=20))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.status == "completed" and result.usage.model_calls == 8
    assert result.usage.action_repairs == 1 and result.usage.tool_calls == 6
    assert "call_tool" not in model.requests[-1].action_schema["discriminator"]["mapping"]


@pytest.mark.parametrize("limit", [1, 2])
def test_question_never_creates_a_wait_with_no_resume_budget(limit):
    model = Scripted(*[{'type': 'ask_user', 'question': 'Laptop model?', 'required_fields': ['laptop_model']}] * limit)
    runtime = AgentRuntime(AgentDependencies(model, demo_tools()), RunPolicy(max_model_calls=limit))
    result = asyncio.run(runtime.run(demo_request()))
    assert result.status == "incomplete" and result.pending_question is None
    assert result.stop_reason == "clarification_budget_exhausted"
    assert result.message is None, "policy stops must leave wording to the caller's stop_reason messages"


def test_late_answer_can_use_one_reserved_evidence_call_then_propose():
    async def scenario():
        ask = {"type": "ask_user", "question": "Laptop model?", "required_fields": ["laptop_model"]}
        model = Scripted(*[deepcopy(SEARCH) for _ in range(5)], ask, SEARCH, PLAN)
        runtime = AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator()),
                               RunPolicy(max_no_progress_calls=20))
        waiting = await runtime.run(demo_request())
        assert waiting.status == "waiting_user" and waiting.usage.model_calls == 6
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"laptop_model": "45W USB-C PD laptop"})
        assert result.status == "completed" and result.usage.model_calls == 8
        assert result.usage.tool_calls == 6
        assert "call_tool" in model.requests[-2].action_schema["discriminator"]["mapping"]
        assert "call_tool" not in model.requests[-1].action_schema["discriminator"]["mapping"]
    asyncio.run(scenario())


def test_budget_aliases_are_rejected_for_new_questions_but_device_fields_are_allowed():
    with pytest.raises(ValidationError, match="purchase_cap"):
        AskUser(type="ask_user", question="Budget?", required_fields=["max_price_usd"])
    assert AskUser(type="ask_user", question="Laptop?", required_fields=["laptop_model"]).required_fields == ["laptop_model"]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    settings = json.loads((root / "configs/knowledge.json").read_text())
    snapshots = tmp_path_factory.mktemp("convergence-snapshots")
    sid = settings["snapshot_id"]
    with ZipFile(root / settings["seed"]["archive"]) as archive:
        for suffix in ("catalog.sqlite", "manifest.json", "category_schemas.json"):
            path = snapshots / sid / suffix
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.read(f"{sid}/{suffix}"))
    return snapshots, sid


def real_dependencies(catalog, model):
    from knowledge.service import KnowledgeService
    from knowledge.agent_tools import build_knowledge_tools
    from knowledge.budget_source import KnowledgeOfferCostSource
    kb = KnowledgeService(catalog[0])
    tools, source = build_knowledge_tools(kb), KnowledgeOfferCostSource(kb)
    return AgentDependencies(model, tools, BudgetPlanValidator(source)), source


def close(deps, source):
    deps.tools.get("search_products").executor.close()
    source.close()


def charger_request(sid, constraints=None):
    return AgentRequest(run_id="belkin-regression", conversation_id="belkin-conversation", snapshot_id=sid,
                        message="I need a USB-C charger for my laptop and phone.",
                        constraints=constraints if constraints is not None else {"purchase_cap": "60.00", "currency": "USD"})


VID = "variant_134634b7f8641d70d5f16f98"
OID = "offer_00caf6d39b3984aa77aa7418"
DESCRIPTION = "ev_51ae1900ab17d8e8e0adb4de"
PRICE = "ev_d9b88b331be2c51c264432bf"
FILTERS = {"category": "chargers", "variant_ids": [VID], "currency": "USD", "max_price_minor": 6000,
           "hard_constraints": {"power": {"op": "gte", "value": "65", "unit": "W"},
                                "ports": {"op": "gte", "value": 2}}}
BELKIN_PLAN = {"type": "propose_plan", "plan": {"items": [
    {"offer_id": OID, "quantity": 1, "evidence_ids": [PRICE, DESCRIPTION]},
]}}


def read_tool(name, **arguments):
    return {"type": "call_tool", "tool_name": name, "arguments": arguments}


def test_infeasible_dev_case_can_finish_early_with_a_scoped_catalog_conclusion(catalog):
    from evaluations.catalog import Catalog, ground_truth
    root = Path(__file__).resolve().parents[2]
    case = next(c for c in map(json.loads, (root / "evaluations/cases.jsonl").read_text().splitlines())
                if c["case_id"] == "dev-07")
    independent = Catalog(catalog[0] / catalog[1] / "catalog.sqlite")
    try:
        assert ground_truth(case, independent)["expected"] == "no_feasible_plan"
    finally:
        independent.close()
    model = Scripted(
        read_tool("search_products", filters={"category": "keyboards", "currency": "USD",
                  "max_price_minor": 1000, "include_unknown": False}),
        {"type": "report_incomplete", "reason": "no_feasible_plan",
         "message": "No verifiable keyboard under 10 USD before tax in this catalog. Unknown prices cannot establish affordability."})
    deps, source = real_dependencies(catalog, model)
    try:
        runtime = AgentRuntime(deps, RunPolicy(max_context_chars=36000))
        result = asyncio.run(runtime.run(AgentRequest(run_id="dev07", conversation_id="dev07",
            snapshot_id=catalog[1], message=case["message"], constraints=case["constraints"])))
        assert result.status == "incomplete" and result.stop_reason == "no_feasible_plan"
        assert result.usage.model_calls == 2 and result.usage.tool_calls == 1
        assert result.verified_plan is None and "10 USD" in result.message
        assert '"total_count": 0' in model.requests[-1].messages[-1].content
    finally:
        close(deps, source)


def test_belkin_unknown_ports_and_power_disagreement_can_use_text_and_pass_real_budget(catalog):
    async def scenario():
        model = Scripted(
            read_tool("describe_filters", category="chargers"),
            read_tool("search_products", filters={**FILTERS, "include_unknown": False}, limit=1),
            read_tool("search_products", filters={**FILTERS, "include_unknown": True}, limit=1),
            read_tool("resolve_evidence", evidence_ids=[DESCRIPTION]),
            read_tool("get_offer_facts", offer_ids=[OID]), BELKIN_PLAN)
        deps, source = real_dependencies(catalog, model)
        # 本用例验证候选和证据摘要，显式留足上下文空间，避免依赖策略提示的长度边界。
        runtime = AgentRuntime(deps, RunPolicy(max_context_chars=20000))
        try:
            result = await runtime.run(charger_request(catalog[1]))
            assert result.status == "completed" and result.usage.model_calls == 6, result.model_dump()
            assert result.usage.tool_calls == 5 and result.verified_plan.total_cost == "43.55"
            final = model.requests[-1]
            memory = next(json.loads(m.content)["shopping_context"] for m in final.messages
                          if m.role == "user" and m.content.startswith('{"shopping_context":'))
            candidate = next(c for c in memory["candidates"] if c["variant_id"] == VID)
            assert candidate["hard_constraint_status"]["ports"] == "unknown"
            assert candidate["hard_constraint_status"]["power"] == "satisfied"
            assert next(a for a in candidate["attributes"] if a["name"] == "power")["status"] == "unknown"
            text = json.dumps(memory["evidence"])
            assert "45W" in text and "20W" in text and DESCRIPTION in text
            assert "default_budget_basis" in final.messages[0].content
            assert '"tax_counted": false' in final.messages[0].content
            assert "raw description" in final.messages[0].content.lower()
            trace = await runtime.export_trace(result.run_id)
            assert trace["shopping_context"]["notes"]
            assert any(e.get("tool") == "resolve_evidence" for e in trace["entries"])
            assert any(e["node"] == "validate_plan" for e in trace["entries"])
        finally:
            close(deps, source)
    asyncio.run(scenario())


def test_budget_preflight_asks_canonical_fields_without_spending_a_model_call(catalog):
    async def scenario():
        model = Scripted(read_tool("search_products", filters={**FILTERS, "include_unknown": True}, limit=1), BELKIN_PLAN)
        deps, source = real_dependencies(catalog, model)
        runtime = AgentRuntime(deps)
        try:
            waiting = await runtime.run(charger_request(catalog[1], {}))
            assert waiting.status == "waiting_user" and waiting.usage.model_calls == 0
            assert waiting.pending_question.required_fields == ["purchase_cap", "currency"]
            result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                          {"purchase_cap": "60.00", "currency": "USD"})
            assert result.status == "completed" and result.usage.model_calls == 2
        finally:
            close(deps, source)
    asyncio.run(scenario())


@pytest.mark.parametrize("currency,expected", [(None, "completed"), ("EUR", "waiting_user")])
def test_old_checkpoint_budget_alias_is_normalized_without_resetting_policy(currency, expected):
    async def scenario():
        saver = InMemorySaver()
        ask = {"type": "ask_user", "question": "Budget?", "required_fields": ["purchase_cap"]}
        first = AgentRuntime(AgentDependencies(Scripted(ask), demo_tools(), MockPlanValidator()), checkpointer=saver)
        request = demo_request().model_copy(update={"constraints": {} if currency is None else {"currency": currency}})
        waiting = await first.run(request)
        config = {"configurable": {"thread_id": f"agent:{waiting.run_id}", "checkpoint_ns": ""}}
        saved = await saver.aget_tuple(config)
        checkpoint = deepcopy(saved.checkpoint)
        values = checkpoint["channel_values"]
        for name in ("pending_question", "action", "clarification"):
            values[name]["required_fields"] = ["max_price_usd"]
        for key in ("shopping_context", "progress", "trace"):
            values.pop(key, None)  # 模拟新增字段之前的检查点。
        for key in ("finalize_reserve_model_calls", "max_no_progress_calls", "max_trace_entries"):
            values["policy"].pop(key, None)
        await saver.aput(config, checkpoint, saved.metadata, checkpoint["channel_versions"])
        model = Scripted(SEARCH, PLAN)
        resumed = AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator()),
                               RunPolicy(max_model_calls=100), checkpointer=saver)
        result = await resumed.resume(waiting.run_id, waiting.pending_question.question_id, {"max_price_usd": "100.00"})
        assert result.status == expected
        if expected == "completed":
            assert result.usage.model_calls == 3
            constraints = json.loads(model.requests[0].messages[1].content)["constraints"]
            assert constraints == {"purchase_cap": "100.00", "currency": "USD"}
            assert '"model_calls_remaining": 7' in model.requests[0].messages[0].content
        else:
            assert result.pending_question.required_fields == ["purchase_cap", "currency"]
            assert result.issues[-1].code == "conflicting_budget_answer" and not model.requests
    asyncio.run(scenario())


def test_evidence_survives_resume_and_context_history_trimming(catalog):
    async def scenario():
        saver = InMemorySaver()
        ask = {"type": "ask_user", "question": "Laptop model and simultaneous charging?",
               "required_fields": ["laptop_model", "simultaneous_charging"]}
        deps, source = real_dependencies(catalog, Scripted(
            read_tool("search_products", filters={**FILTERS, "include_unknown": True}, limit=1),
            read_tool("resolve_evidence", evidence_ids=[DESCRIPTION]), ask))
        first = AgentRuntime(deps, RunPolicy(max_history_messages=1), checkpointer=saver)
        try:
            waiting = await first.run(charger_request(catalog[1]))
            before = await first.export_trace(waiting.run_id)
            model = Scripted(BELKIN_PLAN)
            deps2, source2 = real_dependencies(catalog, model)
            second = AgentRuntime(deps2, RunPolicy(max_model_calls=100), checkpointer=saver)
            try:
                result = await second.resume(waiting.run_id, waiting.pending_question.question_id,
                                              {"laptop_model": "45W USB-C PD laptop", "simultaneous_charging": True})
                assert result.status == "completed" and result.usage.model_calls == 4
                assert result.usage.tool_calls == 2  # 恢复不会重跑搜索。
                assert DESCRIPTION in model.requests[0].model_dump_json() and "45W" in model.requests[0].model_dump_json()
                after = await second.export_trace(waiting.run_id)
                assert len(after["entries"]) > len(before["entries"])
                assert not after["shopping_context"]["candidates"][0].get("hard_constraint_status")
            finally:
                close(deps2, source2)
        finally:
            close(deps, source)
    asyncio.run(scenario())


def test_trace_is_bounded_and_unknown_runs_are_reported():
    async def scenario():
        runtime = AgentRuntime(AgentDependencies(Scripted(SEARCH, PLAN), demo_tools(), MockPlanValidator()),
                               RunPolicy(max_trace_entries=3))
        result = await runtime.run(demo_request())
        trace = await runtime.export_trace(result.run_id)
        assert trace["request"]["message"] == demo_request().message
        assert trace["policy"]["max_model_calls"] == 8
        assert len(trace["entries"]) == 3 and trace["entries"][-1]["node"] == "finalize"
        trace["usage"]["model_calls"] = 99
        assert (await runtime.export_trace(result.run_id))["usage"]["model_calls"] == 2
        with pytest.raises(RunStateError) as exc:
            await runtime.export_trace("missing")
        assert exc.value.code == "run_not_found"
    asyncio.run(scenario())


def test_explicit_usd_alias_can_fill_a_missing_currency_but_not_change_an_existing_one():
    from agent.budget_fields import normalize_legacy_answer
    assert normalize_legacy_answer({"max_price_usd": "80.00"}, {"currency": None}) == {
        "purchase_cap": "80.00", "currency": "USD"}
    with pytest.raises(ValueError, match="currency"):
        normalize_legacy_answer({"max_price_usd": "80.00"}, {"currency": "EUR"})


@pytest.mark.parametrize("default_basis,constraints,basis,tax_counted", [
    ("pre_tax", {}, "pre_tax", False),
    ("all_in", {}, "all_in", True),
    ("pre_tax", {"budget_basis": "all_in"}, "all_in", True),
    ("all_in", {"budget_basis": "pre_tax"}, "pre_tax", False),
])
def test_actual_validator_policy_includes_the_request_override(default_basis, constraints, basis, tax_counted):
    validator = BudgetPlanValidator(None, default_basis=default_basis, require_verified_identity=True)
    policy = validator.decision_policy(constraints)
    assert policy["budget_basis"] == basis and policy["default_budget_basis"] == default_basis
    assert policy["tax_counted"] is tax_counted
    assert policy["require_verified_identity"] is True
