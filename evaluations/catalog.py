"""Independent catalog enumerator (Task 6): ground truth for evaluation cases from the frozen snapshot.

Reads the snapshot's SQLite file directly, with its own code, so that scoring does not depend on the
system's retrieval, tools or validator. For one case it classifies every offer of the case's category:

- feasible:      cost known and within budget, every hard requirement known and satisfied
- over_budget:   cost known and above budget
- violates:      some hard requirement is known and not satisfied
- unverifiable:  cost or some required attribute is unknown (and the offer is not already ruled out)

Budget basis (provisional evaluation rule, see evaluations/README.md): in this snapshot tax is unknown for
every offer, so a strict "all fees known" rule would leave no case with a verifiable plan. Each case states
its basis; "pre_tax" means total = price x quantity + shipping, and an offer with unknown shipping cannot be
shown to be affordable.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

CHECK_PASS, CHECK_FAIL, CHECK_UNKNOWN = "pass", "fail", "unknown"


@dataclass
class Offer:
    offer_id: str
    variant_id: str
    title: str
    merchant: str
    currency: str | None
    price_minor: int | None
    shipping_status: str
    shipping_minor: int | None
    cost_conflict: int
    evidence_id: str


@dataclass
class OfferVerdict:
    offer_id: str
    variant_id: str
    status: str                          # feasible | over_budget | violates | unverifiable
    total: str | None = None             # exact decimal text, when known
    reasons: list[str] = field(default_factory=list)


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def snapshot_id(self) -> str:
        return self.db_path.parent.name

    def offers(self, category: str) -> list[Offer]:
        rows = self.conn.execute(
            "SELECT o.offer_id, o.variant_id, v.title, o.merchant, r.currency, r.price_minor, r.shipping_status,"
            " r.shipping_minor, r.cost_conflict, r.evidence_id"
            " FROM offers o JOIN variants v USING(variant_id) JOIN current_offer_rows r USING(offer_id)"
            " WHERE v.category = ? ORDER BY o.offer_id", (category,)).fetchall()
        return [Offer(**dict(r)) for r in rows]

    def attributes(self, variant_id: str) -> dict:
        rows = self.conn.execute(
            "SELECT canonical_name, status, value_type, bool_value, numeric_value, unit, text_value"
            " FROM current_attribute_rows WHERE variant_id = ?", (variant_id,)).fetchall()
        out = {}
        for r in rows:                    # a known value wins over an unknown one for the same attribute
            if r["canonical_name"] not in out or r["status"] == "known":
                out[r["canonical_name"]] = dict(r)
        return out

    def offer_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT offer_id FROM offers")}

    def offer_category(self, offer_id: str) -> str | None:
        row = self.conn.execute("SELECT v.category FROM offers o JOIN variants v USING(variant_id)"
                                " WHERE o.offer_id = ?", (offer_id,)).fetchone()
        return row[0] if row else None

    def evidence_ids(self, ids) -> set[str]:
        ids = list(ids)
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        return {r[0] for r in self.conn.execute(f"SELECT evidence_id FROM evidence WHERE evidence_id IN ({marks})", ids)}


def check_requirement(attrs: dict, req: dict) -> str:
    """One hard requirement against a variant's current attributes: pass, fail or unknown."""
    a = attrs.get(req["attribute"])
    if a is None or a["status"] != "known":
        return CHECK_UNKNOWN
    op, want = req["op"], req["value"]
    if op == "eq" and isinstance(want, bool):
        if a["bool_value"] is None:
            return CHECK_UNKNOWN
        return CHECK_PASS if bool(a["bool_value"]) == want else CHECK_FAIL
    if op in ("gte", "lte"):
        if a["numeric_value"] is None:
            return CHECK_UNKNOWN
        if req.get("unit") and a.get("unit") and req["unit"].lower() != str(a["unit"]).lower():
            return CHECK_UNKNOWN                 # no silent unit conversion
        value = Decimal(str(a["numeric_value"]))
        ok = value >= Decimal(str(want)) if op == "gte" else value <= Decimal(str(want))
        return CHECK_PASS if ok else CHECK_FAIL
    if op in ("contains_any", "contains_none"):
        text = (a["text_value"] or "").lower()
        if not text:
            return CHECK_UNKNOWN
        hit = any(str(w).lower() in text for w in want)
        return CHECK_PASS if hit == (op == "contains_any") else CHECK_FAIL
    raise ValueError(f"unsupported requirement op {op!r}")


def offer_total(offer: Offer, case: dict) -> Decimal | None:
    """Exact total in major units under the case's budget basis, or None when it cannot be established."""
    if offer.currency != case["constraints"].get("currency") or offer.cost_conflict or offer.price_minor is None:
        return None
    if case.get("budget_basis", "pre_tax") != "pre_tax":
        raise ValueError("only the pre_tax basis is defined; tax is unknown for every offer in this snapshot")
    if offer.shipping_status != "known" or offer.shipping_minor is None:
        return None
    qty = int(case.get("quantity", 1))
    return (Decimal(offer.price_minor) * qty + Decimal(offer.shipping_minor)) / 100


def judge_offer(offer: Offer, attrs: dict, case: dict) -> OfferVerdict:
    v = OfferVerdict(offer.offer_id, offer.variant_id, "feasible")
    cap = case["constraints"].get("purchase_cap")
    total = offer_total(offer, case)
    unknown = False
    if total is None:
        unknown = True
        v.reasons.append("cost_unknown")
    else:
        v.total = str(total)
        if cap is not None and total > Decimal(str(cap)):
            v.status = "over_budget"
            v.reasons.append("over_budget")
    for req in case.get("requirements", []):
        result = check_requirement(attrs, req)
        if result == CHECK_FAIL:
            v.status = "violates" if v.status == "feasible" else v.status
            v.reasons.append(f"violates:{req['attribute']}")
        elif result == CHECK_UNKNOWN:
            unknown = True
            v.reasons.append(f"unknown:{req['attribute']}")
    if v.status == "feasible" and unknown:
        v.status = "unverifiable"
    return v


def ground_truth(case: dict, catalog: Catalog) -> dict:
    """Expected outcome of a case: 'plan' (a verifiable feasible offer exists), 'no_feasible_plan' or
    'clarify' (a required input such as the budget is missing)."""
    if case["constraints"].get("purchase_cap") is None:
        return {"case_id": case["case_id"], "expected": "clarify", "required_fields": ["purchase_cap"],
                "snapshot_id": catalog.snapshot_id()}
    verdicts = [judge_offer(o, catalog.attributes(o.variant_id), case) for o in catalog.offers(case["category"])]
    by_status = {}
    for v in verdicts:
        by_status.setdefault(v.status, []).append(v)
    feasible = sorted(by_status.get("feasible", []), key=lambda v: Decimal(v.total))
    return {
        "case_id": case["case_id"], "snapshot_id": catalog.snapshot_id(),
        "expected": "plan" if feasible else "no_feasible_plan",
        "counts": {k: len(v) for k, v in sorted(by_status.items())},
        "feasible": [{"offer_id": v.offer_id, "variant_id": v.variant_id, "total": v.total} for v in feasible],
        "cheapest_feasible_total": feasible[0].total if feasible else None,
    }


def judge_plan(plan_items: list[dict], case: dict, catalog: Catalog) -> dict:
    """Independent check of a delivered plan (used by the scorer). Single-offer plans are the common case;
    several items are summed, each with its own shipping."""
    known = catalog.offer_ids()
    out = {"unknown_offer_ids": [], "wrong_category": [], "items": [], "total": None,
           "over_budget": False, "violates": False, "unverifiable": False}
    total = Decimal(0)
    for item in plan_items:
        oid = item["offer_id"]
        if oid not in known:
            out["unknown_offer_ids"].append(oid)
            continue
        if catalog.offer_category(oid) != case["category"]:
            out["wrong_category"].append(oid)
        offer = next(o for o in catalog.offers(catalog.offer_category(oid)) if o.offer_id == oid)
        per_item = {**case, "quantity": item.get("quantity", 1)}
        verdict = judge_offer(offer, catalog.attributes(offer.variant_id), per_item)
        out["items"].append({"offer_id": oid, "status": verdict.status, "total": verdict.total,
                             "reasons": verdict.reasons})
        if verdict.total is None:
            out["unverifiable"] = True
        else:
            total += Decimal(verdict.total)
        if any(r.startswith("violates:") for r in verdict.reasons):
            out["violates"] = True
        if any(r.startswith("unknown:") for r in verdict.reasons):
            out["unverifiable"] = True
    cap = case["constraints"].get("purchase_cap")
    if out["items"] and not out["unverifiable"] and not out["unknown_offer_ids"]:
        out["total"] = str(total)
        out["over_budget"] = cap is not None and total > Decimal(str(cap))
    return out
