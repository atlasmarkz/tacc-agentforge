"""Live search on the cloud host (app/live_search.py): the provider, the tool, and the budget validator's view of live offers.
A local stub stands in for SerpApi, so no quota is spent."""
import asyncio
import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pydantic", minversion="2")

# Entries in the shape of real Google Shopping `shopping_results` (field names from the archived responses).
RESULTS = [
    {"title": "Balinese Kintamani Arabica Coffee Beans 250g", "source": "Bali Direct", "extracted_price": 12.5, "price": "$12.50",
     "delivery": "Free delivery", "rating": 4.6, "reviews": 312, "product_link": "https://shop.example/coffee",
     "thumbnail": "https://img.example/coffee.jpg"},
    {"title": "Hand-carved Wooden Mask", "source": "Craft Co", "extracted_price": 30.0, "delivery": "$6.99 delivery",
     "product_link": "https://shop.example/mask"},
    {"title": "Woven Rattan Bag", "source": "Bag House", "extracted_price": 22.0, "product_link": "https://shop.example/bag"},
    {"title": "Free delivery over $35 batik scarf", "source": "Textile Hub", "extracted_price": 18.0, "delivery": "Free delivery over $35"},
    {"title": "No price here", "source": "X"},
]


class Stub:
    """A stand-in SerpApi: counts the calls and can reject chosen keys."""

    def __init__(self, reject=()):
        self.calls, self.reject = [], set(reject)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = parse_qs(urlsplit(self.path).query)
                key = query["api_key"][0]
                stub.calls.append((key, query["q"][0]))
                if key in stub.reject:
                    body, status = {"error": "Your account has run out of searches."}, 200
                else:
                    body, status = {"shopping_results": RESULTS}, 200
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/search.json"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def test_delivery_text_gives_shipping_only_when_it_says_so_plainly():
    from app.live_search import parse_shipping
    assert parse_shipping("Free delivery") == "0.00"
    assert parse_shipping("Free delivery by Sat, Nov 22") == "0.00"
    assert parse_shipping("$6.99 delivery") == "6.99"
    assert parse_shipping("Free delivery over $35") is None          # conditional
    assert parse_shipping("Free delivery on orders $35+") is None
    assert parse_shipping("Delivery in 3-5 days") is None
    assert parse_shipping(None) is None


def test_results_become_offers_with_stable_ids_and_unusable_ones_are_dropped():
    from app.live_search import offer_from_result
    offers = [offer_from_result(r, "2026-11-19T00:00:00+00:00", "bali") for r in RESULTS]
    assert offers[-1] is None                                         # no price
    coffee = offers[0]
    assert coffee["offer_id"].startswith("live_") and coffee["variant_id"].startswith("livev_")
    assert coffee["evidence_id"].startswith("lev_") and coffee["price"] == "12.50" and coffee["shipping"] == "0.00"
    assert offers[2]["shipping"] is None
    assert offer_from_result(RESULTS[0], "later", "other words")["offer_id"] == coffee["offer_id"]


def test_search_rotates_past_a_used_up_key_and_caches_repeats(stub, tmp_path):
    from app.live_search import LiveSearch
    stub.reject = {"k1"}
    live = LiveSearch(["k1", "k2"], daily_limit=10, home=tmp_path, base_url=stub.url)
    offers = live.search("Bali coffee", limit=3)
    assert [k for k, _ in stub.calls] == ["k1", "k2"] and len(offers) == 3
    live.search("bali  coffee")                                       # same words: served from the cache
    assert len(stub.calls) == 2
    assert live.search("Bali coffee", max_price=20)[0]["price"] == "12.50"
    assert all(Decimal(o["price"]) <= 20 for o in live.search("Bali coffee", max_price=20))


def test_the_daily_cap_holds_across_restarts_and_keys_never_leak(stub, tmp_path):
    from app.live_search import LiveSearch, LiveSearchError
    live = LiveSearch(["secretkey1"], daily_limit=2, home=tmp_path, base_url=stub.url, today=lambda: "2026-11-19")
    live.search("one")
    live.search("two")
    with pytest.raises(LiveSearchError) as err:
        live.search("three")
    assert err.value.code == "live_search_limit" and len(stub.calls) == 2
    again = LiveSearch(["secretkey1"], daily_limit=2, home=tmp_path, base_url=stub.url, today=lambda: "2026-11-19")
    with pytest.raises(LiveSearchError):
        again.search("four")                                          # the count was kept on disk
    tomorrow = LiveSearch(["secretkey1"], daily_limit=2, home=tmp_path, base_url=stub.url, today=lambda: "2026-11-20")
    assert tomorrow.search("five")
    stub.reject = {"secretkey1"}
    broken = LiveSearch(["secretkey1"], daily_limit=9, base_url=stub.url)
    with pytest.raises(LiveSearchError) as err:
        broken.search("six")
    assert "secretkey1" not in str(err.value) and "secretkey1" not in err.value.public_message


def test_without_keys_there_is_no_live_search():
    from app.live_search import LiveSearch
    assert LiveSearch.from_env(environ={}) is None
    live = LiveSearch.from_env(environ={"SERPAPI_API_KEYS": "a, b,,a", "SERPAPI_DAILY_LIMIT": "7"})
    assert live.keys == ["a", "b"] and live.daily_limit == 7


def test_the_tool_answers_the_model_and_the_validator_checks_a_live_plan(stub, tmp_path):
    from agent import AgentRequest
    from agent.budget import BudgetPlanValidator, OfferCosts
    from agent.contracts import CallContext, PlanItem, PlanValidationRequest, ProposedPlan
    from app.live_search import LiveOfferCostSource, LiveSearch, live_tool_spec

    class Catalog:                                                   # the catalog side knows nothing about live IDs
        async def load(self, snapshot_id, offer_ids, evidence_ids, context):
            assert not any(i.startswith(("live_", "lev_")) for i in [*offer_ids, *evidence_ids])
            return OfferCosts(snapshot_id=snapshot_id, offers={}, evidence_variants={})

    live = LiveSearch(["k1"], daily_limit=5, home=tmp_path, base_url=stub.url)
    spec = live_tool_spec(live)
    context = CallContext(run_id="r1", snapshot_id="snap", operation_id="r1:1", timeout_s=20.0,
                          cancellation=asyncio.Event(), on_delta=None)

    async def scenario():
        raw = await spec.executor.execute(spec.arguments_model(query="Bali coffee and crafts"), context)
        result = spec.result_model.model_validate(raw)
        by_title = {o.title: o for o in result.offers}
        coffee, mask, bag = (by_title["Balinese Kintamani Arabica Coffee Beans 250g"], by_title["Hand-carved Wooden Mask"],
                             by_title["Woven Rattan Bag"])
        assert (coffee.shipping, mask.shipping, bag.shipping) == ("free", "6.99", "unknown")
        validator = BudgetPlanValidator(LiveOfferCostSource(Catalog(), live.store))
        request = AgentRequest(run_id="r1", conversation_id="c1", snapshot_id="snap", message="Bali gifts",
                               constraints={"purchase_cap": "80.00", "currency": "USD", "budget_basis": "pre_tax"})

        async def check(*items):
            plan = ProposedPlan(items=[PlanItem(offer_id=o.offer_id, quantity=q, evidence_ids=[o.evidence_id]) for o, q in items])
            return await validator.validate(PlanValidationRequest(request=request, plan=plan, proposal_fingerprint="f"), context)

        good = await check((coffee, 2), (mask, 1))
        assert good.status == "passed", good
        assert Decimal(str(good.verified_plan.total_cost)) == Decimal("61.99")      # 2 x 12.50 + 30.00 + 6.99 shipping
        bad = await check((bag, 1))                                  # unknown shipping cannot be verified
        assert bad.status == "failed" and any(i.code == "shipping_unknown" for i in bad.issues)
        ghost = await check((coffee, 1), (type(coffee)(**{**coffee.model_dump(), "offer_id": "live_deadbeef0000", "evidence_id": "lev_deadbeef0000"}), 1))
        assert ghost.status == "failed" and any(i.code == "unknown_offer" for i in ghost.issues)
    asyncio.run(scenario())


def test_the_page_gets_live_offers_in_the_catalog_shape(stub, tmp_path):
    from app.live_search import LiveSearch, lookup_rows
    live = LiveSearch(["k1"], home=tmp_path, base_url=stub.url)
    coffee = live.search("coffee")[0]
    rows = lookup_rows(live.store, [coffee["offer_id"], "live_missing00000"], [coffee["evidence_id"]])
    offer = rows["offers"][0]
    assert offer["store_title"] == coffee["title"] and offer["price_minor"] == 1250 and offer["shipping_status"] == "known"
    assert offer["live"] is True and rows["photos"][coffee["variant_id"]] == "https://img.example/coffee.jpg"
    assert rows["shops"][coffee["offer_id"]] == {"merchant": "Bali Direct", "url": "https://shop.example/coffee"}
    kinds = {e["kind"] for e in rows["evidence"] if e["variant_id"] == coffee["variant_id"]}
    assert kinds == {"offer", "attribute"}                           # a rating/delivery fact the narrative can use
    assert rows["unknown_offer_ids"] == ["live_missing00000"]


def test_the_service_offers_the_tool_only_when_live_search_is_on(stub, tmp_path):
    from agent import RunPolicy
    from app.harness_service import HarnessService, prepare_kb
    from app.live_search import LiveSearch
    from app.selftest import ScriptedPort
    root = tmp_path / "kb"
    root.mkdir()
    from pathlib import Path
    config = Path(__file__).resolve().parents[1] / "configs/knowledge.json"
    snapshot_id, _, kb, _ = prepare_kb(config, root, root)

    def build(live):
        return HarnessService(model=ScriptedPort(), kb=kb, snapshot_id=snapshot_id, policy=RunPolicy(max_context_chars=36000),
                              token="t", info={"job_id": "t"}, explain_default=False, live=live)
    plain, with_live = build(None), build(LiveSearch(["k1"], home=tmp_path, base_url=stub.url))
    try:
        assert plain.tools.get("search_live") is None and plain.health()["live_search"] == {"enabled": False}
        assert with_live.tools.get("search_live") is not None and with_live.tools.get("search_products") is not None
        assert with_live.health()["live_search"]["keys"] == 1
        coffee = with_live.live.search("coffee")[0]
        look = asyncio.run(with_live.lookup({"offer_ids": [coffee["offer_id"]], "evidence_ids": [coffee["evidence_id"]]}))
        assert look["offers"][0]["live"] and look["unknown_offer_ids"] == [] and look["photos"]
    finally:
        plain.close()
        with_live.close()
