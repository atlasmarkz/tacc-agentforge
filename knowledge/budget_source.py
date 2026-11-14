"""知识库侧的 OfferCostSource：为预算校验器读取报价费用和证据归属；查询走现有有界、可取消的工作线程。

只读且不选择价格：同一报价有多个上下文时全部交给校验器，由它判断是否一致。
"""

import time
from dataclasses import replace
from decimal import Decimal

from pydantic import ValidationError

from agent.budget import OfferCost, OfferCosts
from agent.contracts import CapabilityError
from .agent_tools import (
    GetOfferFactsArguments, GetOfferFactsResult, KnowledgeToolExecutor,
    ResolveEvidenceArguments, ResolveEvidenceResult, _Workers,
)
from .store import MINOR_DIGITS


def _money(minor, currency):
    if currency not in MINOR_DIGITS or type(minor) is not int or minor < 0:
        return None
    return format(Decimal(minor).scaleb(-MINOR_DIGITS[currency]), "f")


def offer_cost(row) -> OfferCost:
    """把一行 get_offer_facts 的最小货币单位结果换成主单位字符串；缺失或异常的值记为未知。"""
    currency = row.get("currency")
    price = _money(row.get("price_minor"), currency)
    if price is None:
        currency = None
    known = lambda name: row.get(name + "_status") == "known"  # noqa: E731
    scope = row.get("shipping_scope")
    includes = row.get("price_includes_tax")
    return OfferCost(
        offer_id=row["offer_id"], variant_id=row["variant_id"], price_evidence_id=row["evidence_id"],
        currency=currency, unit_price=price,
        shipping=_money(row.get("shipping_minor"), currency) if known("shipping") else None,
        shipping_scope=scope if scope in ("item", "order") else None,
        tax=_money(row.get("tax_minor"), currency) if known("tax") else None,
        price_includes_tax=includes if includes in ("included", "excluded") else None,
        cost_conflict=bool(row.get("cost_conflict")), variant_conflict=bool(row.get("variant_conflict")),
        identity_verified=row.get("identity_status") == "verified",
    )


class KnowledgeOfferCostSource:
    """宿主在停机时调用 close；不注册成模型可调用的工具。"""

    def __init__(self, service):
        self.workers = _Workers()
        self.offers = KnowledgeToolExecutor(service, "get_offer_facts", self.workers)
        self.evidence = KnowledgeToolExecutor(service, "resolve_evidence", self.workers)

    def close(self):
        self.workers.close()

    async def load(self, snapshot_id, offer_ids, evidence_ids, context):
        if snapshot_id != context.snapshot_id:
            raise CapabilityError("invalid_snapshot", "Offer snapshot does not match the run.")
        deadline = time.monotonic() + context.timeout_s

        async def query(executor, arguments, result_type):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CapabilityError("call_timeout", "Offer cost lookup timed out.")
            raw = await executor.execute(arguments, replace(context, timeout_s=remaining, on_delta=None))
            try:
                result = result_type.model_validate(raw)
            except ValidationError as exc:
                raise CapabilityError("invalid_offer_costs", "Offer service returned invalid data.") from exc
            if result.snapshot_id != snapshot_id:
                raise CapabilityError("invalid_offer_costs", "Offer service answered for another snapshot.")
            return result

        offers: dict[str, list[OfferCost]] = {}
        pending = list(dict.fromkeys(offer_ids))
        while pending:
            page = await query(self.offers, GetOfferFactsArguments(offer_ids=pending), GetOfferFactsResult)
            try:
                for row in page.offers:
                    offers.setdefault(row["offer_id"], []).append(offer_cost(row))
            except (KeyError, ValidationError) as exc:
                raise CapabilityError("invalid_offer_costs", "Offer facts could not be normalized.") from exc
            following = page.remaining_offer_ids
            if following and not set(following) < set(pending):
                raise CapabilityError("invalid_offer_costs", "Offer pagination did not make progress.")
            pending = following

        variants: dict[str, str] = {}
        unique = list(dict.fromkeys(evidence_ids))
        for start in range(0, len(unique), 100):
            batch, cursor, seen = unique[start:start + 100], None, set()
            for _ in range(64):
                page = await query(self.evidence, ResolveEvidenceArguments(evidence_ids=batch, cursor=cursor),
                                   ResolveEvidenceResult)
                for record in page.evidence:
                    eid, owner = record.get("evidence_id"), record.get("variant_id")
                    if eid not in batch or not isinstance(owner, str):
                        raise CapabilityError("invalid_offer_costs", "Evidence service returned an unrelated record.")
                    variants.setdefault(eid, owner)
                cursor = page.next_cursor
                # 未知 ID 不会出现在结果中；没有后续页时，缺失的 ID 就是未知证据。
                if cursor is None or cursor in seen or all(eid in variants for eid in batch):
                    break
                seen.add(cursor)
            else:
                raise CapabilityError("evidence_budget", "Evidence pagination exceeded its bounded budget.")
        return OfferCosts(snapshot_id=snapshot_id, offers=offers, evidence_variants=variants)
