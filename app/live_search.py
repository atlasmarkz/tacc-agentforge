"""Live product search for the cloud deployment: a `search_live` tool backed by SerpApi Google Shopping.

The catalog is a frozen snapshot (that is what keeps the evaluation reproducible), so it cannot serve a goal like
"coffee and handicrafts for a trip to Bali". On the cloud host the agent also gets `search_live`: it asks Google
Shopping at query time and gets back real offers with a price, a seller, a link and a photo.

What stays the same as for catalog offers (agent/ is not touched):
- the budget validator still decides what a plan costs. Live offers get temporary IDs (`live_*` offers, `livev_*`
  products, `lev_*` evidence) and `LiveOfferCostSource` answers for them from this process's memory, so the same
  validator that checks catalog offers checks them, with the same rules: an offer whose delivery cost is unknown
  ("unknown shipping") cannot be put into a plan, and tax is never guessed;
- nothing in a live result is trusted beyond what the search returned: no price is filled in, no delivery cost is
  assumed. "Free delivery" and an amount in the delivery text are the only shipping facts taken from it.

Safety of the quota: keys come from the environment (never from the repository or the model), a daily cap bounds
what one day can spend, results are cached for a while, and keys are redacted from every message that leaves here.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import ssl
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import Field

from agent.budget import OfferCost, OfferCosts
from agent.contracts import CapabilityError, Identifier, Schema
from agent.tools import ToolRegistry, ToolSpec

BASE_URL = "https://serpapi.com/search.json"
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
OFFER_PREFIX, VARIANT_PREFIX, EVIDENCE_PREFIX = "live_", "livev_", "lev_"
DEFAULT_DAILY_LIMIT = 30
TOOL_NAME = "search_live"

TOOL_DESCRIPTION = (
    "Search the live web (Google Shopping) for products when the catalog has nothing suitable or the shopper wants "
    "something it does not carry, such as souvenirs, local specialities or other categories. Returns real offers "
    "with price, seller and delivery cost; use an offer_id and its evidence_id in the plan exactly as given. An offer "
    "whose shipping is 'unknown' cannot be used in a plan: choose another. Each call spends a limited quota, so make "
    "specific queries and do not repeat one. Prefer catalog tools when the catalog covers the need.")


class LiveSearchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.public_message = code, message


def redact(text: str, keys) -> str:
    for key in keys:
        if key:
            text = text.replace(key, "REDACTED")
    return re.sub(r"(api_key=)[^&\"\s]+", r"\1REDACTED", text)


# ---- turning a Google Shopping result into an offer ---------------------------------------------------------------

_FREE = re.compile(r"\bfree\b", re.I)
_CONDITIONAL = re.compile(r"\b(over|on|orders?|min(imum)?|with|when)\b|\+", re.I)
_FEE = re.compile(r"[$]\s?(\d{1,4}(?:\.\d{1,2})?)\s*(?:delivery|shipping)|(?:delivery|shipping)[^$\d]{0,12}[$]\s?(\d{1,4}(?:\.\d{1,2})?)", re.I)


def parse_shipping(text) -> str | None:
    """Shipping amount in dollars from the delivery text, or None when the text does not say plainly."""
    if not isinstance(text, str) or not text.strip():
        return None
    if _FREE.search(text) and not _CONDITIONAL.search(text.replace("Free delivery by", "").replace("Free shipping by", "")):
        return "0.00"
    match = _FEE.search(text)
    if match and not _CONDITIONAL.search(text):
        return f"{Decimal(match.group(1) or match.group(2)):.2f}"
    return None


def _web_url(value) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://")) and len(value) <= 2048


def _clean(text, limit) -> str:
    return " ".join(str(text or "").split())[:limit]


def offer_from_result(item: dict, observed_at: str, query: str) -> dict | None:
    """One live offer from a `shopping_results` entry, or None when it has no usable price or title."""
    title = _clean(item.get("title"), 200)
    price = item.get("extracted_price")
    if not title or isinstance(price, bool) or not isinstance(price, (int, float)) or not 0 < price < 100000:
        return None
    merchant = _clean(item.get("source"), 80) or None
    link = item.get("product_link") or item.get("link")
    digest = hashlib.sha1(json.dumps([title, merchant, round(float(price), 2)], ensure_ascii=False).encode()).hexdigest()[:12]
    rating, reviews = item.get("rating"), item.get("reviews")
    delivery = _clean(item.get("delivery"), 120) or None
    return {
        "offer_id": OFFER_PREFIX + digest, "variant_id": VARIANT_PREFIX + digest, "evidence_id": EVIDENCE_PREFIX + digest,
        "title": title, "merchant": merchant, "currency": "USD", "price": f"{Decimal(str(price)):.2f}",
        "shipping": parse_shipping(delivery), "delivery_text": delivery,
        "rating": float(rating) if isinstance(rating, (int, float)) and not isinstance(rating, bool) and 0 <= rating <= 5 else None,
        "reviews": int(reviews) if isinstance(reviews, int) and not isinstance(reviews, bool) and reviews >= 0 else None,
        "thumbnail": item.get("thumbnail") if _web_url(item.get("thumbnail")) else None,
        "url": link if _web_url(link) else None, "observed_at": observed_at, "query": _clean(query, 160),
    }


def _minor(amount: str | None):
    return None if amount is None else int(Decimal(amount) * 100)


class LiveStore:
    """The live offers returned recently, by offer ID, so the budget validator and the page can look them up."""

    def __init__(self, ttl_s: float = 6 * 3600, max_offers: int = 2000, clock=time.monotonic):
        self.ttl_s, self.max_offers, self.clock = ttl_s, max_offers, clock
        self._items: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def put(self, offers) -> None:
        now = self.clock()
        with self._lock:
            for offer in offers:
                self._items.pop(offer["offer_id"], None)
                self._items[offer["offer_id"]] = (now + self.ttl_s, offer)
            for key in [k for k, (until, _) in self._items.items() if until < now]:
                del self._items[key]
            while len(self._items) > self.max_offers:
                del self._items[next(iter(self._items))]

    def get(self, offer_id: str) -> dict | None:
        with self._lock:
            entry = self._items.get(offer_id)
            return entry[1] if entry and entry[0] >= self.clock() else None

    def by_evidence(self, evidence_id: str) -> dict | None:
        if not evidence_id.startswith(EVIDENCE_PREFIX):
            return None
        return self.get(OFFER_PREFIX + evidence_id[len(EVIDENCE_PREFIX):])


# ---- the provider --------------------------------------------------------------------------------------------------

class LiveSearch:
    def __init__(self, keys, *, daily_limit: int = DEFAULT_DAILY_LIMIT, home: Path | None = None, store: LiveStore | None = None,
                 base_url: str = BASE_URL, timeout_s: float = 12.0, cache_s: float = 900.0, clock=time.monotonic, today=None):
        self.keys = [k for k in dict.fromkeys(k.strip() for k in keys) if k]
        self.daily_limit, self.base_url, self.timeout_s, self.cache_s = daily_limit, base_url, timeout_s, cache_s
        self.store = store or LiveStore()
        self.clock = clock
        self.today = today or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self.counter = (Path(home) / "live-search-count.json") if home else None
        self._lock = threading.Lock()
        self._next_key = 0
        self._used = {"day": self.today(), "used": 0}
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self.last_error: str | None = None                  # the latest provider failure, redacted (shown by /health)
        self._ssl = ssl.create_default_context(cafile=CA_BUNDLE if Path(CA_BUNDLE).exists() else None)
        if self.counter and self.counter.is_file():
            try:
                saved = json.loads(self.counter.read_text(encoding="utf-8"))
                if saved.get("day") == self._used["day"] and isinstance(saved.get("used"), int):
                    self._used = {"day": saved["day"], "used": saved["used"]}
            except (OSError, ValueError):
                pass

    @classmethod
    def from_env(cls, home: Path | None = None, environ=os.environ) -> "LiveSearch | None":
        """None unless SERPAPI_API_KEYS (comma-separated) is set, so a host without keys has no live search at all."""
        keys = [k for k in re.split(r"[,\s]+", environ.get("SERPAPI_API_KEYS", "")) if k]
        if not keys:
            return None
        try:
            limit = int(environ.get("SERPAPI_DAILY_LIMIT") or DEFAULT_DAILY_LIMIT)
        except ValueError:
            limit = DEFAULT_DAILY_LIMIT
        return cls(keys, daily_limit=max(0, limit), home=home)

    def status(self) -> dict:
        with self._lock:
            self._roll()
            return {"enabled": True, "keys": len(self.keys), "used_today": self._used["used"], "daily_limit": self.daily_limit,
                    "last_error": self.last_error}

    def _roll(self) -> None:
        today = self.today()
        if self._used["day"] != today:
            self._used = {"day": today, "used": 0}

    def _reserve(self) -> None:
        """Count one provider call before it is sent; the cap holds even if the call then fails."""
        with self._lock:
            self._roll()
            if self._used["used"] >= self.daily_limit:
                raise LiveSearchError("live_search_limit", "Live search has reached its daily limit; use the catalog instead.")
            self._used["used"] += 1
            if self.counter:
                try:
                    self.counter.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.counter.with_name(self.counter.name + ".tmp")
                    tmp.write_text(json.dumps(self._used), encoding="utf-8")
                    os.replace(tmp, self.counter)
                except OSError:
                    pass

    def _get(self, key: str, query: str) -> dict:
        params = {"engine": "google_shopping", "q": query, "gl": "us", "hl": "en", "api_key": key}
        request = Request(self.base_url + "?" + urlencode(params), headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s, context=self._ssl) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            finally:
                exc.close()
            hint = " ".join(body[:100].split())
            if exc.code in (401, 403, 429):
                raise LiveSearchError("key_rejected", f"HTTP {exc.code} {hint}") from None
            raise LiveSearchError("live_search_unavailable", f"HTTP {exc.code} {hint}") from None
        try:
            data = json.loads(body)
        except ValueError:
            raise LiveSearchError("live_search_unavailable", "not JSON") from None
        if not isinstance(data, dict):
            raise LiveSearchError("live_search_unavailable", "unexpected response")
        error = str(data.get("error") or "")
        if error:
            if "hasn't returned any results" in error:
                return {"shopping_results": []}
            if re.search(r"run out of searches|invalid api key|limit", error, re.I):
                raise LiveSearchError("key_rejected", error[:120])
            raise LiveSearchError("live_search_unavailable", error[:120])
        return data

    def search(self, query: str, limit: int = 6, max_price: float | None = None) -> list[dict]:
        """Live offers for a query (blocking; run it in a worker thread). Raises LiveSearchError."""
        query = _clean(query, 160)
        if not query:
            raise LiveSearchError("invalid_query", "The query is empty.")
        cache_key = query.lower()
        with self._lock:
            hit = self._cache.get(cache_key)
        if hit and hit[0] >= self.clock():
            offers = hit[1]
        else:
            if not self.keys:
                raise LiveSearchError("live_search_unavailable", "Live search is not configured.")
            data, last = None, None
            for _ in range(len(self.keys)):
                self._reserve()
                with self._lock:
                    key = self.keys[self._next_key % len(self.keys)]
                    self._next_key += 1
                try:
                    data = self._get(key, query)
                    break
                except LiveSearchError as exc:                      # a rejected or used-up key: try the next one
                    last = LiveSearchError(exc.code, redact(exc.public_message, self.keys))
                    self.last_error = f"{last.code}: {last.public_message}"[:200]
                    if exc.code != "key_rejected":
                        break
                except (URLError, TimeoutError, OSError) as exc:
                    last = LiveSearchError("live_search_unavailable", redact(f"network error: {type(exc).__name__}", self.keys))
                    self.last_error = f"{last.code}: {redact(f'{type(exc).__name__}: {exc}', self.keys)}"[:200]
                    break
            if data is None:
                code = "live_search_unavailable"
                raise LiveSearchError(code, f"Live search is unavailable right now ({last.code if last else 'error'}).")
            observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
            offers, seen = [], set()
            for item in data.get("shopping_results") or []:
                offer = offer_from_result(item, observed, query) if isinstance(item, dict) else None
                if offer and offer["offer_id"] not in seen:
                    seen.add(offer["offer_id"])
                    offers.append(offer)
            self.store.put(offers)
            with self._lock:
                self._cache[cache_key] = (self.clock() + self.cache_s, offers)
                for stale in [k for k, (until, _) in self._cache.items() if until < self.clock()]:
                    del self._cache[stale]
        picked = [o for o in offers if max_price is None or Decimal(o["price"]) <= Decimal(str(max_price))]
        return picked[:limit]


# ---- the tool ------------------------------------------------------------------------------------------------------

class SearchLiveArguments(Schema):
    query: str = Field(min_length=2, max_length=160)
    max_price: float | None = Field(default=None, gt=0, le=100000)
    limit: int = Field(default=6, strict=True, ge=1, le=10)


class LiveOfferView(Schema):
    offer_id: Identifier
    evidence_id: Identifier
    title: str
    merchant: str | None
    price: str
    currency: str
    shipping: str                       # "free", an amount in the currency, or "unknown"
    rating: float | None
    reviews: int | None


class SearchLiveResult(Schema):
    query: str
    offers: list[LiveOfferView]
    note: str


def _view(offer: dict) -> dict:
    shipping = offer["shipping"]
    return {"offer_id": offer["offer_id"], "evidence_id": offer["evidence_id"], "title": offer["title"][:140],
            "merchant": offer["merchant"], "price": offer["price"], "currency": offer["currency"],
            "shipping": "unknown" if shipping is None else ("free" if Decimal(shipping) == 0 else shipping),
            "rating": offer["rating"], "reviews": offer["reviews"]}


class LiveSearchExecutor:
    def __init__(self, search: LiveSearch):
        self.search = search

    async def execute(self, arguments, context):
        try:
            offers = await asyncio.wait_for(
                asyncio.to_thread(self.search.search, arguments.query, arguments.limit, arguments.max_price),
                timeout=min(context.timeout_s, 20))
        except LiveSearchError as exc:
            raise CapabilityError(exc.code, exc.public_message, recoverable=True) from None
        except asyncio.TimeoutError:
            raise CapabilityError("live_search_unavailable", "Live search timed out.", recoverable=True) from None
        usable = sum(o["shipping"] is not None for o in offers)
        note = ("Prices are live Google Shopping listings in USD; tax is not included. Offers with shipping 'unknown' "
                "cannot be verified for a plan." if offers else "No live offers matched; try different words.")
        if offers and not usable:
            note += " None of these states its delivery cost, so none can go into a plan: search again with other words."
        return SearchLiveResult(query=arguments.query, offers=[_view(o) for o in offers], note=note).model_dump(mode="json")


def live_tool_spec(search: LiveSearch) -> ToolSpec:
    return ToolSpec(name=TOOL_NAME, description=TOOL_DESCRIPTION, arguments_model=SearchLiveArguments,
                    result_model=SearchLiveResult, executor=LiveSearchExecutor(search), version="1", timeout_s=20,
                    max_result_bytes=8192, read_only=True)


def with_live_tool(registry: ToolRegistry, search: LiveSearch) -> ToolRegistry:
    """The same tools plus `search_live`. The registry is read-only once built, so a new one is made from its specs."""
    specs = [registry.get(d.name) for d in registry.descriptions()]
    return ToolRegistry([*specs, live_tool_spec(search)])


# ---- the validator's view of live offers ---------------------------------------------------------------------------

def live_offer_cost(offer: dict) -> OfferCost:
    return OfferCost(
        offer_id=offer["offer_id"], variant_id=offer["variant_id"], price_evidence_id=offer["evidence_id"],
        currency=offer["currency"], unit_price=offer["price"], shipping=offer["shipping"],
        shipping_scope="item" if offer["shipping"] is not None else None,
        tax=None, price_includes_tax=None, cost_conflict=False, variant_conflict=False, identity_verified=False)


class LiveOfferCostSource:
    """Catalog offers go to the catalog's cost source; live offers are answered from the live store."""

    def __init__(self, base, store: LiveStore):
        self.base, self.store = base, store

    async def load(self, snapshot_id, offer_ids, evidence_ids, context):
        live_offers = [i for i in dict.fromkeys(offer_ids) if i.startswith(OFFER_PREFIX)]
        live_evidence = [i for i in dict.fromkeys(evidence_ids) if i.startswith(EVIDENCE_PREFIX)]
        costs = await self.base.load(snapshot_id, [i for i in offer_ids if i not in live_offers],
                                     [i for i in evidence_ids if i not in live_evidence], context)
        offers, variants = dict(costs.offers), dict(costs.evidence_variants)
        for offer_id in live_offers:
            offer = self.store.get(offer_id)
            if offer is not None:
                offers[offer_id] = [live_offer_cost(offer)]
        for evidence_id in live_evidence:
            offer = self.store.by_evidence(evidence_id)
            if offer is not None:
                variants[evidence_id] = offer["variant_id"]
        return OfferCosts(snapshot_id=snapshot_id, offers=offers, evidence_variants=variants)

    def close(self):
        close = getattr(self.base, "close", None)
        if close:
            close()


# ---- the page's view of live offers --------------------------------------------------------------------------------

def lookup_rows(store: LiveStore, offer_ids, evidence_ids) -> dict:
    """Offer, evidence, photo and shop records in the shape /lookup uses for catalog offers, plus the IDs not found."""
    offers, evidence, photos, shops, unknown_offers, unknown_evidence = [], [], {}, {}, [], []
    wanted = list(dict.fromkeys(offer_ids))
    for evidence_id in dict.fromkeys(evidence_ids):
        offer = store.by_evidence(evidence_id)
        if offer is None:
            unknown_evidence.append(evidence_id)
        elif offer["offer_id"] not in wanted:
            wanted.append(offer["offer_id"])
    for offer_id in wanted:
        offer = store.get(offer_id)
        if offer is None:
            if offer_id in offer_ids:
                unknown_offers.append(offer_id)
            continue
        shipping = _minor(offer["shipping"])
        offers.append({"offer_id": offer_id, "variant_id": offer["variant_id"], "store_title": offer["title"],
                       "merchant": offer["merchant"], "currency": offer["currency"], "price_minor": _minor(offer["price"]),
                       "shipping_minor": shipping, "shipping_status": "known" if shipping is not None else "unknown",
                       "shipping_scope": "item" if shipping is not None else None, "tax_status": "unknown",
                       "observed_at": offer["observed_at"], "evidence_id": offer["evidence_id"], "price_stale": False, "live": True})
        price = f"Seller: {offer['merchant'] or 'unknown'}; item price: ${offer['price']}"
        evidence.append({"evidence_id": offer["evidence_id"], "variant_id": offer["variant_id"], "kind": "offer",
                         "title": offer["title"], "text": price, "source_url": offer["url"], "source_label": offer["merchant"],
                         "observed_at": offer["observed_at"], "scope": "live"})
        facts = []
        if offer["rating"] is not None:
            facts.append(f"Rated {offer['rating']}/5" + (f" from {offer['reviews']} reviews" if offer["reviews"] else ""))
        if offer["delivery_text"]:
            facts.append(f"Delivery: {offer['delivery_text']}")
        for number, text in enumerate(facts, 1):
            evidence.append({"evidence_id": f"{offer['evidence_id']}a{number}", "variant_id": offer["variant_id"], "kind": "attribute",
                             "title": offer["title"], "text": text, "source_url": offer["url"], "source_label": offer["merchant"],
                             "observed_at": offer["observed_at"], "scope": "live"})
        if offer["thumbnail"]:
            photos[offer["variant_id"]] = offer["thumbnail"]
        shops[offer_id] = {"merchant": offer["merchant"], "url": offer["url"]}
    return {"offers": offers, "evidence": evidence, "photos": photos, "shops": shops,
            "unknown_offer_ids": unknown_offers, "unknown_evidence_ids": unknown_evidence}
