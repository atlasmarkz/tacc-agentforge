"""知识库解析真实快照的有界解释事实，不产生预算校验结论。"""

import asyncio

import pytest

pydantic = pytest.importorskip("pydantic")
if int(pydantic.VERSION.split(".")[0]) < 2:
    pytest.skip("needs agent dependencies", allow_module_level=True)

from agent.contracts import CallContext, CapabilityError, PlanItem, VerifiedPlan
from knowledge.explanation import KnowledgeEvidenceResolver
from knowledge.service import KnowledgeService
from tests.test_knowledge_service import fixture_snapshot


def plan(evidence_ids=None):
    return VerifiedPlan(snapshot_id="snap-test", currency="USD", total_cost="50.00", remaining_funds="50.00",
                        items=[PlanItem(offer_id="offer-0", quantity=1,
                                        evidence_ids=evidence_ids or ["price-0", "review-0"])])


def context():
    return CallContext(run_id="explanation", snapshot_id="snap-test", operation_id="explanation:evidence:1",
                       timeout_s=5, cancellation=asyncio.Event())


def test_resolver_maps_snapshot_prices_and_paginates_bounded_excerpts(tmp_path):
    async def scenario():
        service = KnowledgeService(fixture_snapshot(tmp_path)[0], max_result_bytes=4096)
        resolver = KnowledgeEvidenceResolver(service)
        try:
            facts = await resolver.resolve(plan(["price-0", "long-review", "review-0"]), context())
            assert {fact.evidence_id for fact in facts} == {"price-0", "long-review", "review-0"}
            assert all(fact.snapshot_id == "snap-test" and fact.offer_id == "offer-0" for fact in facts)
            assert all(fact.unit_price == "50.00" and fact.currency == "USD" for fact in facts)
            assert all(fact.title == "Model 0" and fact.merchant is None for fact in facts)
            assert all(fact.price_age_seconds > 0 and len(fact.excerpt) <= 1400 for fact in facts)
            assert all(fact.source_url == "https://example.com/product" for fact in facts)
            assert all(fact.scope == "product_family_or_unknown" for fact in facts)
        finally:
            resolver.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("ids,code", [(["missing"], "missing_evidence"), (["review-2"], "invalid_evidence")])
def test_resolver_rejects_unknown_and_other_variant_evidence(tmp_path, ids, code):
    async def scenario():
        resolver = KnowledgeEvidenceResolver(KnowledgeService(fixture_snapshot(tmp_path)[0]))
        try:
            with pytest.raises(CapabilityError) as exc:
                await resolver.resolve(plan(ids), context())
            assert exc.value.issue.code == code
            with pytest.raises(CapabilityError, match="snapshot"):
                await resolver.resolve(plan().model_copy(update={"snapshot_id": "other"}), context())
        finally:
            resolver.close()
    asyncio.run(scenario())


def test_resolver_does_not_select_an_ambiguous_offer_context(tmp_path, monkeypatch):
    async def scenario():
        service = KnowledgeService(fixture_snapshot(tmp_path)[0])
        original = service.get_offer_facts
        def ambiguous(snapshot_id, offer_ids):
            page = original(snapshot_id, offer_ids)
            page["offers"].append({**page["offers"][0], "context_key": "other", "price_minor": 999})
            return page
        monkeypatch.setattr(service, "get_offer_facts", ambiguous)
        resolver = KnowledgeEvidenceResolver(service)
        try:
            facts = await resolver.resolve(plan(), context())
            assert all(f.unit_price is None and f.price_age_seconds is None for f in facts)
        finally:
            resolver.close()
    asyncio.run(scenario())


def test_resolver_cancellation_uses_existing_query_worker_contract(tmp_path):
    async def scenario():
        resolver = KnowledgeEvidenceResolver(KnowledgeService(fixture_snapshot(tmp_path)[0]))
        ctx = context()
        ctx.cancellation.set()
        try:
            with pytest.raises(CapabilityError) as exc:
                await resolver.resolve(plan(), ctx)
            assert exc.value.issue.code == "cancelled"
        finally:
            resolver.close()
    asyncio.run(scenario())
