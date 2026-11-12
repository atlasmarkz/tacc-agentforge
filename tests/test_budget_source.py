"""知识库适配器把真实快照的报价事实交给预算校验器；这里不重复规则矩阵，只验证接线与单位换算。"""

import asyncio
import json

import pytest

pydantic = pytest.importorskip("pydantic")
if int(pydantic.VERSION.split(".")[0]) < 2:
    pytest.skip("needs agent dependencies", allow_module_level=True)

from agent import AgentDependencies, AgentRuntime, ModelReply
from agent.budget import BudgetPlanValidator
from agent.contracts import (
    AgentRequest, CallContext, CapabilityError, PlanItem, PlanValidationRequest, ProposedPlan,
)
from knowledge.agent_tools import build_knowledge_tools
from knowledge.budget_source import KnowledgeOfferCostSource, offer_cost
from knowledge.service import KnowledgeError, KnowledgeService
from tests.test_knowledge_service import fixture_snapshot


def context(cancelled=False):
    event = asyncio.Event()
    if cancelled:
        event.set()
    return CallContext(run_id="budget", snapshot_id="snap-test", operation_id="budget:validate:1", timeout_s=5,
                       cancellation=event)


def request(offer="offer-0", quantity=1, evidence=("price-0", "review-0"), **constraints):
    plan = ProposedPlan(items=[PlanItem(offer_id=offer, quantity=quantity, evidence_ids=list(evidence))])
    return PlanValidationRequest(
        request=AgentRequest(run_id="budget", conversation_id="c", snapshot_id="snap-test", message="buy",
                             constraints={"currency": "USD", "purchase_cap": "100.00", **constraints}),
        plan=plan, proposal_fingerprint="a" * 64)


def validate(tmp_path, req, *, patch=None, ctx=None, **options):
    async def scenario():
        service = KnowledgeService(fixture_snapshot(tmp_path, conflict=bool(options.pop("conflict", False)))[0])
        if patch:
            patch(service)
        source = KnowledgeOfferCostSource(service)
        try:
            return await BudgetPlanValidator(source, **options).validate(req, ctx or context())
        finally:
            source.close()
    return asyncio.run(scenario())


def test_the_source_converts_minor_units_and_reports_what_is_unknown(tmp_path):
    async def scenario():
        source = KnowledgeOfferCostSource(KnowledgeService(fixture_snapshot(tmp_path)[0]))
        try:
            costs = await source.load("snap-test", ["offer-0", "ghost"], ["price-0", "review-0", "review-2", "nope"],
                                      context())
        finally:
            source.close()
        assert set(costs.offers) == {"offer-0"}
        row = costs.offers["offer-0"][0]
        assert (row.unit_price, row.currency, row.shipping, row.price_evidence_id) == ("50.00", "USD", "0.00", "price-0")
        assert row.shipping_scope is None and row.tax is None and row.price_includes_tax is None
        assert not row.identity_verified and not row.cost_conflict
        assert costs.evidence_variants == {"price-0": "v-0", "review-0": "v-0", "review-2": "v-2"}
    asyncio.run(scenario())


def test_offer_cost_treats_malformed_amounts_as_unknown():
    base = dict(offer_id="o", variant_id="v", evidence_id="e", currency="USD", price_minor=1999,
                shipping_status="known", shipping_minor=250, shipping_scope="item",
                tax_status="known", tax_minor=160, price_includes_tax="excluded")
    row = offer_cost(base)
    assert (row.unit_price, row.shipping, row.tax) == ("19.99", "2.50", "1.60")
    for broken in ({"price_minor": -1}, {"price_minor": 19.99}, {"price_minor": None}, {"currency": "XXX"}):
        assert offer_cost({**base, **broken}).unit_price is None
    assert offer_cost({**base, "shipping_status": "unknown"}).shipping is None
    assert offer_cost({**base, "tax_minor": -5}).tax is None


def test_a_pre_tax_plan_passes_end_to_end_on_the_real_snapshot(tmp_path):
    result = validate(tmp_path, request())
    assert result.status == "passed"
    assert (result.verified_plan.total_cost, result.verified_plan.remaining_funds) == ("50.00", "50.00")
    assert result.verified_plan.cost_breakdown == {"items": "50.00", "shipping": "0.00"}


def test_the_snapshot_never_proves_tax_so_the_all_in_basis_asks_instead(tmp_path):
    result = validate(tmp_path, request(), default_basis="all_in")
    assert result.status == "failed" and [i.code for i in result.issues] == ["tax_unknown"]


def test_several_units_need_a_known_shipping_scope(tmp_path):
    result = validate(tmp_path, request(quantity=2), default_basis="pre_tax")
    assert [i.code for i in result.issues] == ["shipping_scope_unknown"]


def test_evidence_about_another_product_is_rejected(tmp_path):
    result = validate(tmp_path, request(evidence=("price-0", "review-2")), default_basis="pre_tax")
    assert [i.code for i in result.issues] == ["foreign_evidence"]


def test_conflicting_price_observations_are_reported(tmp_path):
    result = validate(tmp_path, request(), default_basis="pre_tax", conflict=True)
    assert [i.code for i in result.issues] == ["conflicting_prices"]


def test_the_budget_boundary_uses_the_snapshot_price_exactly(tmp_path):
    assert validate(tmp_path / "a", request(purchase_cap="50.00"), default_basis="pre_tax").status == "passed"
    over = validate(tmp_path / "b", request(purchase_cap="49.99"), default_basis="pre_tax")
    assert [i.code for i in over.issues] == ["budget_exceeded"]


def test_a_cancelled_call_is_unavailable_not_a_budget_verdict(tmp_path):
    result = validate(tmp_path, request(), ctx=context(cancelled=True), default_basis="pre_tax")
    assert result.status == "unavailable" and [i.code for i in result.issues] == ["cancelled"]


def test_a_declared_service_failure_is_unavailable(tmp_path):
    def broken(service):
        def fail(*_, **__):
            raise KnowledgeError("snapshot_changed", "An immutable snapshot changed; publish a new snapshot ID.")
        service.get_offer_facts = fail
    result = validate(tmp_path, request(), patch=broken, default_basis="pre_tax")
    assert result.status == "unavailable" and [i.code for i in result.issues] == ["snapshot_changed"]


def test_a_programming_error_is_not_disguised_as_a_budget_result(tmp_path):
    def broken(service):
        def fail(*_, **__):
            raise RuntimeError("boom")
        service.get_offer_facts = fail
    with pytest.raises(RuntimeError):
        validate(tmp_path, request(), patch=broken, default_basis="pre_tax")


def test_the_source_rejects_a_snapshot_that_is_not_the_runs(tmp_path):
    async def scenario():
        source = KnowledgeOfferCostSource(KnowledgeService(fixture_snapshot(tmp_path)[0]))
        try:
            with pytest.raises(CapabilityError) as exc:
                await source.load("other", ["offer-0"], ["price-0"], context())
            assert exc.value.issue.code == "invalid_snapshot"
        finally:
            source.close()
    asyncio.run(scenario())


class ScriptedModel:
    """按顺序给出预设动作，并记录每次收到的请求。"""

    def __init__(self, *actions):
        self.actions, self.requests = list(actions), []

    async def generate(self, request, context):
        self.requests.append(request)
        return ModelReply(content=json.dumps(self.actions.pop(0)))


def plan_action(quantity):
    return {"type": "propose_plan", "plan": {"items": [
        {"offer_id": "offer-0", "quantity": quantity, "evidence_ids": ["price-0"]}]}}


def run_agent(tmp_path, model, **constraints):
    async def scenario():
        service = KnowledgeService(fixture_snapshot(tmp_path)[0])
        source = KnowledgeOfferCostSource(service)
        try:
            runtime = AgentRuntime(AgentDependencies(
                model, build_knowledge_tools(service), BudgetPlanValidator(source)))
            return await runtime.run(AgentRequest(
                run_id="budget-run", conversation_id="c", snapshot_id="snap-test", message="buy earbuds",
                constraints={"currency": "USD", "purchase_cap": "100.00", **constraints}))
        finally:
            source.close()
    return asyncio.run(scenario())


def test_the_runtime_accepts_the_validator_receipt_and_completes(tmp_path):
    result = run_agent(tmp_path, ScriptedModel(plan_action(1)))
    assert result.status == "completed"
    assert (result.verified_plan.total_cost, result.verified_plan.remaining_funds) == ("50.00", "50.00")


def test_a_rejected_plan_is_explained_to_the_model_which_then_repairs_it(tmp_path):
    model = ScriptedModel(plan_action(2), plan_action(1))
    result = run_agent(tmp_path, model)
    assert result.status == "completed" and result.usage.plan_repairs == 1
    feedback = model.requests[1].messages[-1].content
    assert "shipping_scope_unknown" in feedback and "offer-0" in feedback


def test_a_plan_over_the_budget_never_completes(tmp_path):
    model = ScriptedModel(*[plan_action(1)] * 3)
    result = run_agent(tmp_path, model, purchase_cap="40.00")
    assert result.status != "completed" and result.verified_plan is None
