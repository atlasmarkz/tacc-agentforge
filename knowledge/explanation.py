"""知识库侧的 EvidenceResolver；查询通过现有有界、可取消的工作线程执行。"""

import time
from dataclasses import replace
from decimal import Decimal

from pydantic import ValidationError

from agent.contracts import CapabilityError, EvidenceFact
from .agent_tools import (
    GetOfferFactsArguments, GetOfferFactsResult, KnowledgeToolExecutor,
    ResolveEvidenceArguments, ResolveEvidenceResult, _Workers,
)
from .store import MINOR_DIGITS, clean_url


class KnowledgeEvidenceResolver:
    """宿主在停机时调用 close；不注册成模型可调用的工具。"""

    def __init__(self, service):
        self.workers = _Workers()
        self.offers = KnowledgeToolExecutor(service, "get_offer_facts", self.workers)
        self.evidence = KnowledgeToolExecutor(service, "resolve_evidence", self.workers)

    def close(self):
        self.workers.close()

    async def resolve(self, plan, context):
        if plan.snapshot_id != context.snapshot_id:
            raise CapabilityError("invalid_evidence", "Evidence snapshot does not match the verified plan.")
        deadline = time.monotonic() + context.timeout_s

        async def query(executor, arguments, result_type):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CapabilityError("call_timeout", "Evidence resolution timed out.")
            raw = await executor.execute(arguments, replace(context, timeout_s=remaining, on_delta=None))
            try:
                result = result_type.model_validate(raw)
            except ValidationError as exc:
                raise CapabilityError("invalid_evidence", "Evidence service returned invalid data.") from exc
            if result.snapshot_id != plan.snapshot_id or result.unknown_ids:
                raise CapabilityError("missing_evidence", "Evidence is unavailable in the selected snapshot.")
            return result

        remaining_ids = list(dict.fromkeys(item.offer_id for item in plan.items))
        offer_rows = []
        while remaining_ids:
            page = await query(self.offers, GetOfferFactsArguments(offer_ids=remaining_ids), GetOfferFactsResult)
            offer_rows.extend(page.offers)
            following = page.remaining_offer_ids
            if following and (not set(following) < set(remaining_ids)):
                raise CapabilityError("invalid_evidence", "Offer pagination did not make progress.")
            remaining_ids = following
        ids = list(dict.fromkeys(eid for item in plan.items for eid in item.evidence_ids))
        records = {}
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            cursor = None
            cursors = set()
            for _ in range(64):
                page = await query(self.evidence, ResolveEvidenceArguments(evidence_ids=batch, cursor=cursor),
                                   ResolveEvidenceResult)
                for record in page.evidence:
                    eid = record.get("evidence_id")
                    if eid not in batch:
                        raise CapabilityError("invalid_evidence", "Evidence service returned an unrelated record.")
                    if eid not in records:
                        if record.get("text_start", 0) != 0:
                            raise CapabilityError("invalid_evidence", "The initial evidence excerpt is missing.")
                        records[eid] = record
                if all(eid in records for eid in batch):
                    break  # Only the first bounded excerpt is needed, not the entire archive.
                cursor = page.next_cursor
                if cursor is None or cursor in cursors:
                    raise CapabilityError("missing_evidence", "Evidence pagination is incomplete.")
                cursors.add(cursor)
            else:
                raise CapabilityError("evidence_budget", "Evidence pagination exceeded its bounded budget.")
        facts = []
        for item in plan.items:
            rows = [row for row in offer_rows if row.get("offer_id") == item.offer_id]
            if not rows:
                raise CapabilityError("missing_evidence", "The verified offer is unavailable.")
            titles = {row.get("store_title") for row in rows if row.get("store_title")}
            merchants = {row.get("merchant") for row in rows if row.get("merchant")}
            for eid in dict.fromkeys(item.evidence_ids):
                record = records[eid]
                if any(row.get("variant_id") != record.get("variant_id") for row in rows):
                    raise CapabilityError("invalid_evidence", "Evidence does not belong to the verified offer's variant.")
                # VerifiedPlan has no offer context key. Do not choose an arbitrary price
                # among destination/quantity contexts, or treat a conflicting price as known.
                price = age = currency = None
                if (len(rows) == 1 and not rows[0].get("cost_conflict") and not rows[0].get("variant_conflict")
                        and rows[0].get("currency") == plan.currency and type(rows[0].get("price_minor")) is int
                        and rows[0]["price_minor"] >= 0 and plan.currency in MINOR_DIGITS):
                    price = format(Decimal(rows[0]["price_minor"]).scaleb(-MINOR_DIGITS[plan.currency]), "f")
                    age, currency = rows[0].get("price_age_seconds"), plan.currency
                try:
                    facts.append(EvidenceFact(
                        snapshot_id=plan.snapshot_id, offer_id=item.offer_id, evidence_id=eid,
                        title=next(iter(titles)) if len(titles) == 1 else record.get("title", ""),
                        merchant=next(iter(merchants)) if len(merchants) == 1 else None,
                        unit_price=price, currency=currency, price_age_seconds=age,
                        excerpt=record.get("text", "")[:1400], source_url=clean_url(record.get("source_url")),
                        scope=record.get("scope"),
                    ))
                except ValidationError as exc:
                    raise CapabilityError("invalid_evidence", "Evidence facts could not be normalized.") from exc
        return facts
