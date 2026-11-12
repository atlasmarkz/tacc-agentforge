"""预算校验器：边界、精确算术、两种限额取小、未知费用与证据归属（使用假的报价源）。"""

import asyncio
from decimal import Decimal

import pytest

pydantic = pytest.importorskip("pydantic")
if int(pydantic.VERSION.split(".")[0]) < 2:
    pytest.skip("needs agent dependencies", allow_module_level=True)

from pydantic import ValidationError

from agent.budget import BudgetPlanValidator, OfferCost, OfferCosts
from agent.contracts import (
    AgentRequest, CallContext, CapabilityError, PlanItem, PlanValidationRequest, ProposedPlan,
)

SNAPSHOT = "snap-test"
FINGERPRINT = "f" * 64


def cost(offer="o1", **fields):
    base = dict(offer_id=offer, variant_id="v-" + offer, price_evidence_id="p-" + offer, currency="USD",
                unit_price="10.00", shipping="2.00", shipping_scope="order", price_includes_tax="included")
    return OfferCost(**{**base, **fields})


class FakeSource:
    def __init__(self, *rows, evidence=None, error=None, snapshot=SNAPSHOT):
        self.rows, self.error, self.snapshot = rows, error, snapshot
        self.evidence = evidence
        self.calls = []

    async def load(self, snapshot_id, offer_ids, evidence_ids, context):
        self.calls.append((snapshot_id, offer_ids, evidence_ids))
        if self.error:
            raise self.error
        offers = {}
        for row in self.rows:
            offers.setdefault(row.offer_id, []).append(row)
        evidence = self.evidence if self.evidence is not None else {
            eid: "v-" + eid.split("-", 1)[1] for eid in evidence_ids}
        return OfferCosts(snapshot_id=self.snapshot, offers=offers, evidence_variants=evidence)


def run(source, items=None, constraints=None, **options):
    items = items or [("o1", 1)]
    plan = ProposedPlan(items=[PlanItem(offer_id=o, quantity=q, evidence_ids=["p-" + o]) for o, q in items])
    request = PlanValidationRequest(
        request=AgentRequest(run_id="run-1", conversation_id="c-1", snapshot_id=SNAPSHOT, message="buy",
                             constraints={"currency": "USD", "purchase_cap": "100.00", **(constraints or {})}),
        plan=plan, proposal_fingerprint=FINGERPRINT)
    context = CallContext(run_id="run-1", snapshot_id=SNAPSHOT, operation_id="run-1:validate:1", timeout_s=5,
                          cancellation=asyncio.Event())
    return asyncio.run(BudgetPlanValidator(source, **options).validate(request, context))


def codes(result):
    return [issue.code for issue in result.issues]


def test_passes_with_exact_itemized_total_and_remaining_funds():
    result = run(FakeSource(cost()), [("o1", 3)])
    plan = result.verified_plan
    assert result.status == "passed" and result.proposal_fingerprint == FINGERPRINT
    assert (plan.total_cost, plan.remaining_funds) == ("32.00", "68.00")
    assert plan.cost_breakdown == {"items": "30.00", "shipping": "2.00", "tax": "0.00"}
    assert plan.currency == "USD" and plan.snapshot_id == SNAPSHOT and plan.items[0].quantity == 3


def test_decimal_arithmetic_has_no_float_error():
    result = run(FakeSource(cost(unit_price="0.10", shipping="0.00")), [("o1", 3)], {"purchase_cap": "0.30"})
    assert result.status == "passed"
    assert (result.verified_plan.total_cost, result.verified_plan.remaining_funds) == ("0.30", "0.00")


@pytest.mark.parametrize("cap,status", [("12.00", "passed"), ("11.99", "failed")])
def test_total_equal_to_available_funds_passes_and_one_cent_over_fails(cap, status):
    result = run(FakeSource(cost()), constraints={"purchase_cap": cap})
    assert result.status == status
    if status == "passed":
        assert result.verified_plan.remaining_funds == "0.00"
    else:
        assert codes(result) == ["budget_exceeded"] and result.verified_plan is None


@pytest.mark.parametrize("quantity", [0, -1])
def test_zero_and_negative_quantities_cannot_enter_a_plan(quantity):
    with pytest.raises(ValidationError):
        PlanItem(offer_id="o1", quantity=quantity, evidence_ids=["p-o1"])


@pytest.mark.parametrize("constraints,available", [
    ({"purchase_cap": "100", "periodic_budget": "120", "spending": "30", "reserve": "10"}, "80.00"),
    ({"purchase_cap": "60", "periodic_budget": "120", "spending": "30", "reserve": "10"}, "60"),
    ({"periodic_budget": "50"}, "50"),
    ({"purchase_cap": "50"}, "50"),
])
def test_available_funds_are_the_smaller_of_the_cap_and_the_remaining_balance(constraints, available):
    result = run(FakeSource(cost()), constraints={"purchase_cap": None, **constraints})
    assert result.status == "passed"
    assert Decimal(result.verified_plan.remaining_funds) == Decimal(available) - Decimal("12.00")


def test_a_negative_remaining_balance_leaves_nothing_affordable():
    result = run(FakeSource(cost()), constraints={"purchase_cap": None, "periodic_budget": "10", "reserve": "20"})
    assert result.status == "failed" and codes(result) == ["budget_exceeded"]


@pytest.mark.parametrize("constraints,code", [
    ({"purchase_cap": None}, "missing_budget"),
    ({"purchase_cap": "-1"}, "invalid_budget"),
    ({"purchase_cap": "abc"}, "invalid_budget"),
    ({"purchase_cap": True}, "invalid_budget"),
    ({"purchase_cap": "NaN"}, "invalid_budget"),
    ({"purchase_cap": None, "spending": "5"}, "incomplete_periodic_budget"),
    ({"currency": None}, "missing_currency"),
    ({"currency": "usd"}, "missing_currency"),
    ({"budget_basis": "rough"}, "invalid_budget_basis"),
])
def test_missing_or_invalid_inputs_fail_with_a_field_to_ask_for(constraints, code):
    result = run(FakeSource(cost()), constraints=constraints)
    assert result.status == "failed" and code in codes(result)
    assert result.verified_plan is None


def test_budget_and_cost_problems_are_reported_together():
    result = run(FakeSource(cost(shipping=None)), constraints={"purchase_cap": None})
    assert set(codes(result)) == {"missing_budget", "shipping_unknown"}


def test_a_different_currency_is_never_converted():
    result = run(FakeSource(cost(currency="EUR")))
    assert codes(result) == ["currency_mismatch"]


def test_zero_decimal_currencies_keep_whole_amounts():
    result = run(FakeSource(cost(currency="JPY", unit_price="1500", shipping="300")), [("o1", 2)],
                 {"currency": "JPY", "purchase_cap": "4000"})
    assert (result.verified_plan.total_cost, result.verified_plan.remaining_funds) == ("3300", "700")


@pytest.mark.parametrize("fields,quantity,shipping", [
    ({"shipping_scope": "item"}, 3, "6.00"),
    ({"shipping_scope": "order"}, 3, "2.00"),
    ({"shipping_scope": None}, 1, "2.00"),
])
def test_shipping_follows_its_scope(fields, quantity, shipping):
    result = run(FakeSource(cost(**fields)), [("o1", quantity)])
    assert result.verified_plan.cost_breakdown["shipping"] == shipping


def test_shipping_of_unknown_scope_is_unknown_for_several_units():
    assert codes(run(FakeSource(cost(shipping_scope=None)), [("o1", 2)])) == ["shipping_scope_unknown"]


def test_unknown_shipping_blocks_an_affordability_claim():
    result = run(FakeSource(cost(shipping=None)))
    assert result.status == "failed" and codes(result) == ["shipping_unknown"]


@pytest.mark.parametrize("fields", [
    {"price_includes_tax": None},
    {"price_includes_tax": "excluded"},
    {"price_includes_tax": None, "tax": "1.00"},
])
def test_all_in_basis_does_not_guess_the_tax(fields):
    assert codes(run(FakeSource(cost(**fields)), default_basis="all_in")) == ["tax_unknown"]


def test_the_default_basis_is_pre_tax_and_a_request_can_ask_for_all_in():
    unknown_tax = cost(price_includes_tax=None)
    assert run(FakeSource(unknown_tax)).status == "passed"
    assert codes(run(FakeSource(unknown_tax), constraints={"budget_basis": "all_in"})) == ["tax_unknown"]


@pytest.mark.parametrize("how", ["default", "constraint"])
def test_pre_tax_basis_counts_listed_price_and_shipping_only(how):
    row = cost(price_includes_tax=None)
    if how == "default":
        result = run(FakeSource(row))
    else:
        result = run(FakeSource(row, ), default_basis="all_in", constraints={"budget_basis": "pre_tax"})
    assert result.status == "passed" and result.verified_plan.total_cost == "12.00"
    assert "tax" not in result.verified_plan.cost_breakdown


def test_known_tax_is_added_per_unit_when_the_price_excludes_it():
    result = run(FakeSource(cost(price_includes_tax="excluded", tax="0.80")), [("o1", 2)])
    plan = result.verified_plan
    assert plan.cost_breakdown == {"items": "20.00", "shipping": "2.00", "tax": "1.60"}
    assert plan.total_cost == "23.60"


def test_several_items_are_summed_exactly():
    rows = [cost("a", unit_price="19.99"), cost("b", unit_price="5.02", shipping="0.00")]
    result = run(FakeSource(*rows), [("a", 2), ("b", 1)])
    assert result.verified_plan.cost_breakdown["items"] == "45.00"
    assert result.verified_plan.total_cost == "47.00"


@pytest.mark.parametrize("fields,code", [
    ({"cost_conflict": True}, "conflicting_prices"),
    ({"variant_conflict": True}, "variant_conflict"),
    ({"unit_price": None}, "price_unknown"),
    ({"currency": None}, "price_unknown"),
])
def test_conflicting_or_missing_prices_do_not_pass(fields, code):
    assert codes(run(FakeSource(cost(**fields)))) == [code]


def test_an_offer_with_different_costs_per_context_is_ambiguous():
    rows = [cost(), cost(unit_price="11.00")]
    assert codes(run(FakeSource(*rows))) == ["ambiguous_offer_context"]
    assert run(FakeSource(cost(), cost())).status == "passed"


def test_unverified_identity_only_blocks_in_strict_mode():
    assert run(FakeSource(cost())).status == "passed"
    assert codes(run(FakeSource(cost()), require_verified_identity=True)) == ["identity_unverified"]
    assert run(FakeSource(cost(identity_verified=True)), require_verified_identity=True).status == "passed"


def test_unknown_and_duplicate_offers_fail():
    assert codes(run(FakeSource(cost()), [("o1", 1), ("ghost", 1)])) == ["unknown_offer"]
    plan_items = [("o1", 1), ("o1", 2)]
    assert "duplicate_offer" in codes(run(FakeSource(cost()), plan_items))


def test_price_and_other_evidence_must_belong_to_the_chosen_offer():
    assert run(FakeSource(cost(), evidence={"p-o1": "v-o1"})).status == "passed"
    assert codes(run(FakeSource(cost(), evidence={"p-o1": "v-other"}))) == ["foreign_evidence"]
    assert codes(run(FakeSource(cost(), evidence={}))) == ["unknown_evidence"]
    assert codes(run(FakeSource(cost(price_evidence_id="other-price"), evidence={"p-o1": "v-o1"}))) == ["price_evidence_missing"]


def test_source_failures_are_unavailable_not_a_rule_violation():
    result = run(FakeSource(error=CapabilityError("cancelled", "Knowledge query was cancelled.")))
    assert result.status == "unavailable" and codes(result) == ["cancelled"]
    assert run(FakeSource(cost(), snapshot="other")).status == "unavailable"


def test_the_source_is_asked_only_for_the_plans_offers_and_evidence():
    source = FakeSource(cost("a"), cost("b"))
    run(source, [("a", 1), ("b", 1)])
    assert source.calls == [(SNAPSHOT, ["a", "b"], ["p-a", "p-b"])]


def test_at_most_twenty_issues_are_returned():
    rows = [cost(f"o{i}", shipping=None, price_includes_tax=None) for i in range(12)]
    result = run(FakeSource(*rows), [(f"o{i}", 1) for i in range(12)], default_basis="all_in")
    assert result.status == "failed" and len(result.issues) == 20


def test_the_default_basis_must_be_known():
    with pytest.raises(ValueError):
        BudgetPlanValidator(FakeSource(), default_basis="rough")
