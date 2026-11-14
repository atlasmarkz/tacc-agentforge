"""精确预算校验器（Task 3）：金额只用 Decimal 计算，不交给模型。

校验器实现 ``PlanValidator`` 端口，不导入知识库；报价事实由注入的 ``OfferCostSource`` 提供
（知识库侧适配器见 ``knowledge/budget_source.py``）。

返回状态的含义（运行时据此决定下一步）：
- ``passed``：附带 ``VerifiedPlan``，其中的金额都由本模块计算；
- ``failed``：规则问题，模型能据此换方案或向用户追问（每个问题的 message 会点名需要的字段）；
- ``unavailable``：报价事实读取失败等基础设施问题，运行会终止。

预算输入（``AgentRequest.constraints``，金额用十进制字符串，也接受整数和浮点）：
- ``currency``：三位大写币种，必填，不做换算；
- ``purchase_cap``：单次购买上限；
- ``periodic_budget`` 减去 ``spending`` 和 ``reserve``：周期预算的余额（后两项默认 0）；
- 两种限额同时给出时，可用资金取较小者；
- ``budget_basis``：``all_in`` 或 ``pre_tax``，缺省时使用校验器的默认值。

规则要点：
- 运费未知、币种不一致、价格冲突、规格版本冲突或证据不属于所选商品，一律不通过，不估算；
- 口径决定税费是否计入，对方案中的每个条目都一样（不会因为某个条目的税费未知而改变其他条目的算法）：
  - ``pre_tax``（默认，项目已决定按不含税比较）：只计标价和已知运费，税费一律不计入（已知也不计），
    ``cost_breakdown`` 不含 ``tax``，面向用户的回答必须注明不含税；
  - ``all_in``：每个条目都计税：标价含税计 0，标价不含税且税费已知时按单价同样的方式乘数量；
    税费未知或不知道标价是否含税就不通过（``tax_unknown``），``cost_breakdown`` 总是含 ``tax``；
- 运费按 ``shipping_scope`` 计算：``item`` 乘数量，``order`` 每个方案条目计一次（假设每个条目是一次订单，
  商家合并运费时是偏高的上界）；范围未知时只有数量为 1 才能确定；
- 总价恰好等于可用资金时通过；
- 价格新鲜度不在这里判断（冻结快照按快照时点重放），SKU 身份默认不阻止通过，可用 ``require_verified_identity`` 收紧。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Literal, Protocol

from pydantic import Field

from .contracts import (
    Amount, CallContext, CapabilityError, Identifier, Issue, PlanValidationRequest, Schema,
    ValidationResult, VerifiedPlan,
)

BudgetBasis = Literal["all_in", "pre_tax"]
_MAX_ISSUES = 20


class OfferCost(Schema):
    """一个报价观测的费用事实；金额为主单位的十进制字符串，None 表示未知。"""

    offer_id: Identifier
    variant_id: Identifier
    price_evidence_id: Identifier
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    unit_price: Amount | None = None
    shipping: Amount | None = None
    shipping_scope: Literal["item", "order"] | None = None
    tax: Amount | None = None
    price_includes_tax: Literal["included", "excluded"] | None = None
    cost_conflict: bool = False
    variant_conflict: bool = False
    identity_verified: bool = False


class OfferCosts(Schema):
    """所选快照中的报价费用，以及方案引用的证据分别属于哪个商品。"""

    snapshot_id: Identifier
    offers: dict[str, list[OfferCost]]  # 同一报价可有多个数量或目的地上下文；缺失的键表示未知报价
    evidence_variants: dict[str, str]  # 证据 ID 到商品 ID；缺失的键表示未知证据


class OfferCostSource(Protocol):
    async def load(self, snapshot_id: str, offer_ids: list[str], evidence_ids: list[str],
                   context: CallContext) -> OfferCosts: ...


def _parse_amount(constraints, name, issues, default=None):
    value = constraints.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        issues.append(Issue(code="invalid_budget", message=f"{name} must be a nonnegative decimal amount."))
        return None
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount < 0:
        issues.append(Issue(code="invalid_budget", message=f"{name} must be a nonnegative decimal amount."))
        return None
    return amount


def _funds(constraints, issues):
    """可用资金：购买上限与（周期预算 − 已花费 − 预留）中较小者；信息不足时记录问题并返回 None。"""
    cap = _parse_amount(constraints, "purchase_cap", issues)
    periodic = _parse_amount(constraints, "periodic_budget", issues)
    spending = _parse_amount(constraints, "spending", issues, Decimal(0))
    reserve = _parse_amount(constraints, "reserve", issues, Decimal(0))
    if issues:
        return None
    if periodic is None and (constraints.get("spending") is not None or constraints.get("reserve") is not None):
        issues.append(Issue(code="incomplete_periodic_budget",
                            message="spending and reserve need periodic_budget; ask the user for periodic_budget."))
        return None
    limits = [value for value in (cap, None if periodic is None else periodic - spending - reserve) if value is not None]
    if not limits:
        issues.append(Issue(code="missing_budget",
                            message="No budget is known; ask the user for purchase_cap or periodic_budget."))
        return None
    return min(limits)


def _format(value, digits):
    quantum = Decimal(1).scaleb(-digits)
    rounded = value.quantize(quantum)
    return format(rounded if rounded == value else value, "f")


def _scale(text):
    exponent = Decimal(text).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


class BudgetPlanValidator:
    """用报价事实计算方案总价并与可用资金比较。"""

    def __init__(self, source: OfferCostSource, *, default_basis: BudgetBasis = "pre_tax",
                 require_verified_identity: bool = False):
        if default_basis not in ("all_in", "pre_tax"):
            raise ValueError("default_basis must be all_in or pre_tax")
        self.source, self.default_basis = source, default_basis
        self.require_verified_identity = require_verified_identity

    async def validate(self, request: PlanValidationRequest, context: CallContext) -> ValidationResult:
        fingerprint = request.proposal_fingerprint
        agent_request, plan = request.request, request.plan
        offer_ids = list(dict.fromkeys(item.offer_id for item in plan.items))
        evidence_ids = list(dict.fromkeys(eid for item in plan.items for eid in item.evidence_ids))
        try:
            costs = await self.source.load(agent_request.snapshot_id, offer_ids, evidence_ids, context)
        except CapabilityError as exc:
            return ValidationResult(status="unavailable", proposal_fingerprint=fingerprint, issues=[exc.issue])
        if costs.snapshot_id != agent_request.snapshot_id:
            issue = Issue(code="snapshot_mismatch", message="Offer costs came from a different snapshot.")
            return ValidationResult(status="unavailable", proposal_fingerprint=fingerprint, issues=[issue])
        with localcontext() as decimal_context:
            decimal_context.prec = 60
            issues, verified = self._evaluate(agent_request, plan, costs)
        if issues:
            return ValidationResult(status="failed", proposal_fingerprint=fingerprint, issues=issues[:_MAX_ISSUES])
        return ValidationResult(status="passed", proposal_fingerprint=fingerprint, verified_plan=verified)

    def _evaluate(self, agent_request, plan, costs):
        constraints = agent_request.constraints
        issues: list[Issue] = []
        available = _funds(constraints, issues)
        currency = constraints.get("currency")
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isupper() \
                or not currency.isalpha():
            issues.append(Issue(code="missing_currency",
                                message="currency must be a three-letter uppercase code; ask the user for currency."))
            currency = None
        basis = constraints.get("budget_basis", self.default_basis)
        if basis not in ("all_in", "pre_tax"):
            issues.append(Issue(code="invalid_budget_basis", message="budget_basis must be all_in or pre_tax."))
            basis = self.default_basis
        if len({item.offer_id for item in plan.items}) != len(plan.items):
            issues.append(Issue(code="duplicate_offer", message="List each offer once and merge its quantities."))

        items_total = shipping_total = tax_total = Decimal(0)
        all_in, digits = basis == "all_in", 0
        for item in plan.items:
            problems = []
            rows = costs.offers.get(item.offer_id) or []
            if not rows:
                issues.append(Issue(code="unknown_offer", message=f"Offer {item.offer_id} is not in the snapshot."))
                continue
            row = rows[0]
            if any(r.cost_conflict for r in rows):
                problems.append(("conflicting_prices", "has conflicting price observations"))
            elif len({(r.currency, r.unit_price, r.shipping, r.shipping_scope)
                      + ((r.tax, r.price_includes_tax) if all_in else ()) for r in rows}) > 1:
                problems.append(("ambiguous_offer_context", "has different costs for different quantity or destination contexts"))
            if any(r.variant_conflict for r in rows):
                problems.append(("variant_conflict", "has a store title that conflicts with its product variant"))
            if self.require_verified_identity and not all(r.identity_verified for r in rows):
                problems.append(("identity_unverified", "has an unverified product identity"))
            if not {r.price_evidence_id for r in rows} & set(item.evidence_ids):
                problems.append(("price_evidence_missing", "must cite its price evidence"))
            for eid in item.evidence_ids:
                owner = costs.evidence_variants.get(eid)
                if owner is None:
                    problems.append(("unknown_evidence", f"cites evidence {eid} that is not in the snapshot"))
                elif owner != row.variant_id:
                    problems.append(("foreign_evidence", f"cites evidence {eid} about a different product"))
            if row.unit_price is None or row.currency is None:
                problems.append(("price_unknown", "has no known price or currency"))
            elif currency is not None and row.currency != currency:
                problems.append(("currency_mismatch", f"is priced in {row.currency}, not {currency}; no conversion is done"))
            quantity = item.quantity
            shipping = None
            if row.shipping is None:
                problems.append(("shipping_unknown", "has unknown shipping; ask the user or choose another offer"))
            elif row.shipping_scope == "item":
                shipping = Decimal(row.shipping) * quantity
            elif row.shipping_scope == "order" or (row.shipping_scope is None and quantity == 1):
                shipping = Decimal(row.shipping)
            else:
                problems.append(("shipping_scope_unknown", "has shipping that may be per item or per order"))
            tax = Decimal(0)                      # pre_tax: tax is never counted, known or not
            if all_in:                            # included in the price: 0; excluded and known: per unit
                if row.price_includes_tax == "excluded" and row.tax is not None:
                    tax = Decimal(row.tax) * quantity
                elif row.price_includes_tax != "included":
                    problems.append(("tax_unknown", "has unknown tax; ask whether a pre-tax comparison is acceptable"))
            for code, text in problems:
                issues.append(Issue(code=code, message=f"Offer {item.offer_id} {text}."))
            if problems or row.unit_price is None or shipping is None:
                continue
            digits = max(digits, _scale(row.unit_price))
            items_total += Decimal(row.unit_price) * quantity
            shipping_total += shipping
            tax_total += tax
        if issues or available is None or currency is None:
            return issues, None
        total = items_total + shipping_total + tax_total
        if total > available:
            issues.append(Issue(code="budget_exceeded", message=(
                f"Total {_format(total, digits)} {currency} exceeds available funds {_format(available, digits)} {currency}.")))
            return issues, None
        breakdown = {"items": _format(items_total, digits), "shipping": _format(shipping_total, digits)}
        if all_in:                                # the breakdown always tells which basis was used
            breakdown["tax"] = _format(tax_total, digits)
        return [], VerifiedPlan(
            snapshot_id=agent_request.snapshot_id, items=plan.items, currency=currency,
            total_cost=_format(total, digits), remaining_funds=_format(available - total, digits),
            cost_breakdown=breakdown,
        )
