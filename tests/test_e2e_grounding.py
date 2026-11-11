"""check_grounding of the knowledge e2e script (deploy/tacc/service/agent-e2e/e2e_kb.py).

Review of PR #12 found two false positives of the first version, which checked only that IDs exist:
(1) real IDs the model never saw in this run, (2) a real offer with evidence of a different product.
"""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/tacc/service/agent-e2e"))
import e2e_kb  # noqa: E402


class FakeKb:
    """get_offer_facts / resolve_evidence over a tiny catalog, with the real pagination shapes."""

    def __init__(self, offers, evidence, offers_per_page=100):
        self.offers, self.evidence, self.per_page = offers, evidence, offers_per_page    # id -> variant

    def get_offer_facts(self, snapshot_id, offer_ids):
        known = [o for o in offer_ids if o in self.offers]
        page, rest = known[:self.per_page], known[self.per_page:]
        return {"offers": [{"offer_id": o, "variant_id": self.offers[o]} for o in page],
                "unknown_ids": [o for o in offer_ids if o not in self.offers], "remaining_offer_ids": rest}

    def resolve_evidence(self, snapshot_id, evidence_ids, cursor=None):
        start = int(cursor or 0)
        known = [e for e in evidence_ids if e in self.evidence]
        page = known[start:start + 1]                     # one record per page
        nxt = str(start + 1) if start + 1 < len(known) else None
        return {"evidence": [{"evidence_id": e, "variant_id": self.evidence[e]} for e in page],
                "unknown_ids": [e for e in evidence_ids if e not in self.evidence], "next_cursor": nxt}


KB = FakeKb(offers={"off-charger": "var-charger", "off-earbuds": "var-earbuds"},
            evidence={"ev-charger-1": "var-charger", "ev-charger-2": "var-charger", "ev-keyboard": "var-keyboard",
                      "ev-earbuds": "var-earbuds"})


def plan(offer, *evidence):
    return {"items": [{"offer_id": offer, "quantity": 1, "evidence_ids": list(evidence)}]}


def ground(p, context):
    return e2e_kb.check_grounding(KB, "snap", p, context)


def test_grounded_when_ids_exist_were_seen_and_belong_together():
    g = ground(plan("off-charger", "ev-charger-1", "ev-charger-2"), "off-charger ev-charger-1 ev-charger-2")
    assert g["grounded"] is True and g["evidence_of_another_variant"] == []


def test_real_ids_that_the_model_never_saw_are_not_grounded():           # review case 1: no tool call
    g = ground(plan("off-earbuds", "ev-earbuds"), "system prompt and user request only")
    assert g["unknown_offers"] == [] and g["unknown_evidence"] == []
    assert g["offers_seen_in_context"] == [False] and g["evidence_seen_in_context"] == [False]
    assert g["grounded"] is False


def test_only_the_offer_seen_is_not_enough():
    assert ground(plan("off-charger", "ev-charger-1"), "off-charger")["grounded"] is False


def test_evidence_of_another_product_is_not_grounded():                  # review case 2: charger + keyboard
    g = ground(plan("off-charger", "ev-keyboard"), "off-charger ev-keyboard")
    assert g["unknown_offers"] == [] and g["unknown_evidence"] == []
    assert g["evidence_of_another_variant"] == [{"offer_id": "off-charger", "evidence_id": "ev-keyboard"}]
    assert g["grounded"] is False


def test_one_foreign_evidence_among_good_ones_still_fails():
    g = ground(plan("off-charger", "ev-charger-1", "ev-keyboard"), "off-charger ev-charger-1 ev-keyboard")
    assert g["grounded"] is False and len(g["evidence_of_another_variant"]) == 1


def test_invented_ids_are_reported_as_unknown():
    g = ground(plan("off-made-up", "ev-made-up"), "off-made-up ev-made-up")
    assert g["unknown_offers"] == ["off-made-up"] and g["unknown_evidence"] == ["ev-made-up"] and not g["grounded"]


def test_paginated_answers_are_followed_to_the_end():
    # The foreign evidence is on the second page of the evidence answer, the second offer on the second offers page.
    kb = FakeKb(KB.offers, KB.evidence, offers_per_page=1)
    p = {"items": [{"offer_id": "off-charger", "quantity": 1, "evidence_ids": ["ev-charger-1", "ev-keyboard"]},
                   {"offer_id": "off-earbuds", "quantity": 1, "evidence_ids": ["ev-earbuds"]}]}
    g = e2e_kb.check_grounding(kb, "snap", p, "off-charger off-earbuds ev-charger-1 ev-keyboard ev-earbuds")
    assert g["evidence_of_another_variant"] == [{"offer_id": "off-charger", "evidence_id": "ev-keyboard"}]
    assert g["unknown_offers"] == [] and g["grounded"] is False


@pytest.mark.skipif(os.name == "nt" and not sys.flags.utf8_mode,
                    reason="seed install needs UTF-8 mode on Windows (issue #5)")
def test_both_review_cases_on_the_real_seed_snapshot(tmp_path):
    from knowledge.bootstrap import install_snapshot
    from knowledge.service import KnowledgeService
    from knowledge.settings import load_settings
    s = load_settings("configs/knowledge.json", data_root=tmp_path, snapshots_root=tmp_path / "snaps")
    install_snapshot(s)
    kb = KnowledgeService(s.snapshots_root, **s.service)

    def first_offer(category):
        cards = kb.search_products(s.snapshot_id, {"category": category, "currency": "USD"}, limit=5)["candidates"]
        price = next(c["selected_price"] for c in cards if (c.get("selected_price") or {}).get("offer_id"))
        return price["offer_id"], price["evidence_id"]

    (earbuds_offer, earbuds_ev), (charger_offer, charger_ev), (_, keyboard_ev) = (
        first_offer("earbuds"), first_offer("chargers"), first_offer("keyboards"))
    g = lambda offer, ev, ctx: e2e_kb.check_grounding(kb, s.snapshot_id, plan(offer, ev), ctx)   # noqa: E731
    assert not g(earbuds_offer, earbuds_ev, "request only")["grounded"]                       # case 1: no tool call
    assert not g(charger_offer, keyboard_ev, f"{charger_offer} {keyboard_ev}")["grounded"]    # case 2: mismatched
    assert g(charger_offer, charger_ev, f"{charger_offer} {charger_ev}")["grounded"]          # the honest case
