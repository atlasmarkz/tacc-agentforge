"""预算字段约定及旧问题兼容；不从自然语言推断金额或币种。"""

from decimal import Decimal, InvalidOperation


USD_ALIASES = {"max_price_usd", "budget_usd"}
AMOUNT_FIELDS = {"purchase_cap", "periodic_budget", "spending", "reserve"}


def amount_valid(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    try:
        number = Decimal(str(value))
        return number.is_finite() and number >= 0
    except InvalidOperation:
        return False


def normalize_legacy_answer(answer, constraints):
    """仅显式 USD 别名可推导币种；冲突不能静默覆盖。"""
    result = dict(answer)
    aliases = USD_ALIASES.intersection(answer)
    if not aliases:
        return result
    amounts = [answer[name] for name in aliases]
    if any(not amount_valid(value) for value in amounts):
        raise ValueError("Legacy USD budget must be a nonnegative decimal amount.")
    if len({Decimal(str(value)) for value in amounts}) != 1:
        raise ValueError("Legacy USD budget fields conflict.")
    currency = result.get("currency", constraints.get("currency"))
    if currency is not None and currency != "USD":
        raise ValueError("Legacy USD budget conflicts with currency; answer with canonical fields.")
    canonical = result.get("purchase_cap", constraints.get("purchase_cap"))
    if canonical is not None and (not amount_valid(canonical) or Decimal(str(canonical)) != Decimal(str(amounts[0]))):
        raise ValueError("Legacy USD budget conflicts with purchase_cap.")
    for name in aliases:
        result.pop(name)
    result.update(purchase_cap=str(amounts[0]), currency="USD")
    return result
