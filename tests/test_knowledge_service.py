"""Knowledge service correctness, bounds and existing agent-port integration."""
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from knowledge.service import KnowledgeError, KnowledgeService


def put(connection, table, **row):
    columns = list(row)
    connection.execute("INSERT INTO " + table + "(" + ",".join(columns) + ") VALUES(" +
                       ",".join("?" for _ in columns) + ")", [row[k] for k in columns])


def fixture_snapshot(tmp_path, *, conflict=False):
    root = tmp_path / "snapshots"
    folder = root / "snap-test"
    folder.mkdir(parents=True)
    path = folder / "catalog.sqlite"
    connection = sqlite3.connect(path)
    schema = Path(__file__).parents[1] / "knowledge/schema.sql"
    connection.executescript(schema.read_text())
    for key, value in {"snapshot_id": "snap-test", "dataset_id": "dataset-test", "as_of": "2021-01-02T00:00:00+00:00",
                       "schema_version": "shopping-kb-2", "retrieval_version": "fts5-unicode61-v1"}.items():
        put(connection, "metadata", key=key, value=value)
    attributes = {"battery_life": {"type": "number", "unit": "h", "units": {"h": "1", "min": "0.0166666666666666667"}},
                  "anc": {"type": "boolean"}}
    for category in ("earbuds", "backpacks", "unseen-category"):
        put(connection, "category_schemas", category_id=category, schema_json=json.dumps({"attributes": attributes}))
    put(connection, "raw_records", raw_id="raw-1", call_id="call-1", engine="google_product", origin_sha256="a" * 64,
        archive_sha256="b" * 64, archive_path="raw/raw-1.json", observed_at="2021-01-01T00:00:00+00:00", processed_at=None, params_json="{}")

    def evidence(eid, vid, text, kind="review", group=None):
        put(connection, "evidence", evidence_id=eid, variant_id=vid, kind=kind, title="Review" if kind == "review" else "Offer",
            text=text, source_url="https://example.com/product", source_label="Example", raw_id="raw-1",
            json_pointer="/product_results/" + eid, observed_at="2021-01-01T00:00:00+00:00",
            scope="product_family_or_unknown", duplicate_group_id=group)

    for i in range(9):
        vid = f"v-{i}"
        category = "earbuds" if i < 8 else "backpacks"
        put(connection, "variants", variant_id=vid, provider_product_id=f"p-{i}", title=f"Earbuds {i}" if i < 8 else "Waterproof backpack",
            category=category, type="unknown", identity_status="provider_group_unverified", detail_status="details_available",
            source_url="https://example.com/product", currency="USD", currency_basis="explicit", observed_at="2021-01-01T00:00:00+00:00")
        oid = f"offer-{i}"
        put(connection, "offers", offer_id=oid, variant_id=vid, merchant_id="merchant", merchant="Merchant", url="https://example.com/product",
            identity_status="provider_group_unverified", store_title=f"Model {i}", selected_options_json="{}")
        evidence(f"price-{i}", vid, "Item price in USD", "offer")
        ob = dict(observation_id=f"ob-{i}", offer_id=oid, raw_id="raw-1", evidence_id=f"price-{i}",
                  observed_at="2021-01-01T00:00:00+00:00", currency="USD", currency_basis="explicit", price_minor=5000 + i * 1000,
                  shipping_minor=0, shipping_status="known", shipping_scope="unknown", tax_minor=None, tax_status="unknown",
                  price_includes_tax="unknown", shipping_text="Free", additional_costs="+tax", displayed_total_minor=5000 + i * 1000,
                  context_json='{"destination":null,"quantity":null}', context_key="context", cost_signature=f"cost-{i}", condition=None, store_title=f"Model {i}")
        put(connection, "offer_observations", **ob)
        if i == 0:
            evidence("old-price", vid, "Old item price", "offer")
            put(connection, "offer_observations", **{**ob, "observation_id": "ob-old", "evidence_id": "old-price",
                                                    "observed_at": "2020-01-01T00:00:00+00:00", "price_minor": 100,
                                                    "cost_signature": "old-cost"})
            if conflict:
                evidence("conflict-price", vid, "Conflicting item price", "offer")
                put(connection, "offer_observations", **{**ob, "observation_id": "ob-conflict", "evidence_id": "conflict-price",
                                                        "price_minor": 4500, "cost_signature": "conflicting-cost"})
        text = "battery comfort earbuds" if i < 2 else "waterproof battery earbuds" if i < 8 else "waterproof waterproof backpack"
        evidence(f"review-{i}", vid, text, group="duplicate-pair" if i < 2 else f"g-{i}")
        if i != 1:
            evidence(f"attribute-{i}", vid, "Battery life: 0.1 h", "attribute")
            value = "0.1" if i != 2 else "0.05"
            put(connection, "attributes", attribute_id=f"attr-{i}", variant_id=vid, name="battery life", value=value + " h",
                normalized_number=value, unit="h", evidence_id=f"attribute-{i}", canonical_name="battery_life", value_type="number",
                status="known", numeric_value=float(value), bool_value=None, text_value=None, observed_at="2021-01-01T00:00:00+00:00")
    evidence("long-review", "v-0", "中文证据内容。" * 2000)
    for target, source in (("current_offer_rows", "_computed_current_offer_observations"),
                           ("current_attribute_rows", "_computed_current_attributes"),
                           ("current_listing_rows", "_computed_current_listings")):
        connection.execute("INSERT INTO " + target + " SELECT * FROM " + source)
    connection.commit()
    connection.close()
    return root, path


def size(result):
    return len(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def test_compact_catalog_pagination_and_stable_full_scope(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    service = KnowledgeService(root)
    first = service.search_products("snap-test", {"category": "earbuds"}, limit=5)
    second = service.search_products("snap-test", {"category": "earbuds"}, limit=5, cursor=first["next_cursor"])
    assert first["total_count"] == 8 and first["scope_complete"]
    assert first["candidate_set_id"] == second["candidate_set_id"]
    ids = [x["variant_id"] for x in first["candidates"] + second["candidates"]]
    assert len(ids) == len(set(ids)) == 8
    assert first["truncated"] and second["next_cursor"] is None
    assert first["bytes"] == size(first) <= 8192
    assert "tax" in first["candidates"][0]["selected_price"]["unknown_fields"]
    assert first["candidates"][0]["selected_price"]["complete_total_minor"] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_latest_price_inclusive_boundary_and_decimal_constraints(tmp_path):
    root, _ = fixture_snapshot(tmp_path)
    service = KnowledgeService(root)
    result = service.search_products("snap-test", {"variant_ids": ["v-0"], "currency": "USD", "max_price_minor": 5000,
                                                     "hard_constraints": {"battery_life": {"op": "eq", "value": "0.1", "unit": "h"}}})
    assert result["total_count"] == 1
    assert result["candidates"][0]["selected_price"]["observation_id"] == "ob-0"
    assert result["candidates"][0]["hard_constraint_status"]["battery_life"] == "satisfied"
    assert service.search_products("snap-test", {"variant_ids": ["v-0"], "currency": "USD", "max_price_minor": 4999})["total_count"] == 0


def test_unknown_hard_constraint_retained_and_violations_excluded(tmp_path):
    root, _ = fixture_snapshot(tmp_path)
    service = KnowledgeService(root)
    filters = {"variant_ids": ["v-0", "v-1", "v-2"], "category": "earbuds",
               "hard_constraints": {"battery_life": {"op": "gte", "value": "0.1", "unit": "h"}}}
    result = service.search_products("snap-test", filters)
    assert {c["variant_id"]: c["constraint_status"] for c in result["candidates"]} == {"v-0": "satisfied", "v-1": "unknown"}
    result = service.search_products("snap-test", {**filters, "include_unknown": False})
    assert [c["variant_id"] for c in result["candidates"]] == ["v-0"]


def test_conflicting_current_costs_cannot_supply_selected_price(tmp_path):
    root, _ = fixture_snapshot(tmp_path, conflict=True)
    service = KnowledgeService(root)
    result = service.search_products("snap-test", {"variant_ids": ["v-0"]})
    assert result["candidates"][0]["selected_price"] is None
    offers = service.get_offer_facts("snap-test", ["offer-0"])["offers"]
    assert len(offers) == 2
    assert all(o["cost_conflict"] and "cost_conflict" in o["unknown_fields"] and o["complete_total_minor"] is None for o in offers)


def test_explicit_variant_capacity_conflict_preserves_offer_but_not_candidate_price(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE variants SET title='Boundary Waters Backpack 143L' WHERE variant_id='v-0'")
    connection.execute("UPDATE offers SET store_title='Boundary Waters 143 L roll top' WHERE offer_id='offer-0'")
    put(connection, "offers", offer_id="offer-other-size", variant_id="v-0", merchant_id="merchant", merchant="Merchant",
        url="https://example.com/product", identity_status="provider_group_unverified", store_title="Boundary Waters 70L Regular",
        selected_options_json='{"size":"70L"}')
    columns = [r[1] for r in connection.execute("PRAGMA table_info(offer_observations)")]
    row = dict(zip(columns, connection.execute("SELECT * FROM offer_observations WHERE observation_id='ob-0'").fetchone()))
    put(connection, "offer_observations", **{**row, "offer_id": "offer-other-size", "observation_id": "ob-other-size", "price_minor": 1000})
    connection.execute("DELETE FROM current_offer_rows")
    connection.execute("INSERT INTO current_offer_rows SELECT * FROM _computed_current_offer_observations")
    connection.commit(); connection.close()
    service = KnowledgeService(root)
    candidate = service.search_products("snap-test", {"variant_ids": ["v-0"]})["candidates"][0]
    assert candidate["selected_price"]["offer_id"] == "offer-0"
    assert candidate["selected_price"]["price_minor"] == 5000
    facts = service.get_offer_facts("snap-test", ["offer-other-size"])["offers"]
    assert len(facts) == 1 and facts[0]["price_minor"] == 1000
    assert facts[0]["variant_conflict"] and "variant_conflict" in facts[0]["unknown_fields"]
    assert facts[0]["complete_total_minor"] is None


def test_filtered_fts_precedes_top_k_and_modes_share_candidate_scope(tmp_path):
    root, _ = fixture_snapshot(tmp_path)
    service = KnowledgeService(root)
    candidate_set = service.search_products("snap-test", {"variant_ids": ["v-0", "v-1"]})["candidate_set_id"]
    before = service.describe_candidate_set("snap-test", candidate_set)
    filtered = service.retrieve_evidence("snap-test", '耳机 ("battery") OR NOT -(', candidate_set, k=20)
    assert filtered["evidence"] and {e["variant_id"] for e in filtered["evidence"]} <= {"v-0", "v-1"}
    groups = [e["duplicate_group_id"] or e["evidence_id"] for e in filtered["evidence"]]
    assert len(groups) == len(set(groups))
    keyword = service.retrieve_evidence("snap-test", "waterproof", candidate_set, k=20, mode="keyword")
    assert any(e["variant_id"] == "v-8" for e in keyword["evidence"])
    assert service.describe_candidate_set("snap-test", candidate_set) == before
    assert filtered["bytes"] == size(filtered) <= 8192


def test_resolve_unicode_evidence_paginates_without_losing_text(tmp_path):
    root, _ = fixture_snapshot(tmp_path)
    service = KnowledgeService(root, max_result_bytes=2048)
    cursor, pieces = None, []
    for _ in range(100):
        result = service.resolve_evidence("snap-test", ["long-review"], cursor=cursor)
        assert result["bytes"] == size(result) <= 2048
        assert result["evidence"][0]["archive_path"] == "raw/raw-1.json"
        pieces.extend(e["text"] for e in result["evidence"])
        cursor = result["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("evidence pagination did not terminate")
    assert "".join(pieces) == "中文证据内容。" * 2000


def test_retrieval_byte_budget_preserves_all_top_k_ids(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    connection = sqlite3.connect(path)
    for i in range(5):
        connection.execute("UPDATE evidence SET text=?,title=?,duplicate_group_id=NULL WHERE evidence_id=?",
                           ("中文证据内容 " * 1000, "中文标题" * 50, f"review-{i}"))
    connection.commit(); connection.close()
    full = KnowledgeService(root, max_result_bytes=65536)
    full_set = full.search_products("snap-test", {"category": "earbuds"})["candidate_set_id"]
    reference = full.retrieve_evidence("snap-test", "中文证据内容", full_set, k=5)
    bounded = KnowledgeService(root)
    bounded_set = bounded.search_products("snap-test", {"category": "earbuds"})["candidate_set_id"]
    result = bounded.retrieve_evidence("snap-test", "中文证据内容", bounded_set, k=5)
    assert [e["evidence_id"] for e in result["evidence"]] == [e["evidence_id"] for e in reference["evidence"]]
    assert result["returned_count"] == 5 and result["truncated"] and size(result) <= 8192


def test_offer_context_groups_are_atomic_under_byte_cap(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    connection = sqlite3.connect(path)
    columns = [r[1] for r in connection.execute("PRAGMA table_info(offer_observations)")]
    row = dict(zip(columns, connection.execute("SELECT * FROM offer_observations WHERE observation_id='ob-0'").fetchone()))
    for i in range(15):
        put(connection, "offer_observations", **{**row, "observation_id": f"ob-context-{i}", "context_key": f"context-{i}"})
    connection.execute("DELETE FROM current_offer_rows")
    connection.execute("INSERT INTO current_offer_rows SELECT * FROM _computed_current_offer_observations")
    connection.commit(); connection.close()
    with pytest.raises(KnowledgeError) as error:
        KnowledgeService(root).get_offer_facts("snap-test", ["offer-0"])
    assert error.value.code == "result_budget_exceeded"


def test_catalog_at_most_four_initial_selects_and_cache_can_disable(tmp_path, monkeypatch):
    import knowledge.service as module
    root, _ = fixture_snapshot(tmp_path)
    original, traced = sqlite3.connect, []
    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(traced.append)
        return connection
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    service = KnowledgeService(root, cache_size=0)
    first = service.search_products("snap-test", {"category": "earbuds"})
    selects = [s for s in traced if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 4
    traced.clear()
    second = service.search_products("snap-test", {"category": "earbuds"})
    assert len([s for s in traced if s.lstrip().upper().startswith("SELECT")]) == 3
    assert first["candidate_set_id"] == second["candidate_set_id"]
    assert service.retrieve_evidence("snap-test", "battery", first["candidate_set_id"])["evidence"]


def test_cached_price_ages_recomputed(tmp_path, monkeypatch):
    import knowledge.service as module
    root, _ = fixture_snapshot(tmp_path)
    class Clock(datetime):
        current = datetime(2022, 1, 1, tzinfo=timezone.utc)
        @classmethod
        def now(cls, tz=None):
            return cls.current
    monkeypatch.setattr(module, "datetime", Clock)
    service = KnowledgeService(root)
    first = service.search_products("snap-test", {"variant_ids": ["v-0"]})
    Clock.current += timedelta(hours=1)
    second = service.search_products("snap-test", {"variant_ids": ["v-0"]})
    assert first["candidate_set_id"] == second["candidate_set_id"]
    assert second["candidates"][0]["selected_price"]["price_age_seconds"] - first["candidates"][0]["selected_price"]["price_age_seconds"] == 3600


@pytest.mark.parametrize("filters", [{"sql": "DROP TABLE variants"}, {"max_price_minor": 1}, {"currency": "usd"},
                                     {"category": []}, {"category": {}},
                                     {"variant_ids": "v-0"}, {"hard_constraints": {"no-such-field": {"op": "eq", "value": 1}}}])
def test_invalid_filters_are_explicit(tmp_path, filters):
    root, _ = fixture_snapshot(tmp_path)
    with pytest.raises(KnowledgeError, match=".") as error:
        KnowledgeService(root).search_products("snap-test", filters)
    assert error.value.code == "invalid_arguments"
    assert error.value.recoverable is True


def test_snapshot_paths_context_deadlines_and_read_only(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    service = KnowledgeService(root)
    with pytest.raises(KnowledgeError):
        service.describe("../snap-test")
    with service._connection(service._snapshot("snap-test")) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM variants")


def test_deadline_and_cancellation_signals(tmp_path):
    root, _ = fixture_snapshot(tmp_path)
    service = KnowledgeService(root)
    with pytest.raises(KnowledgeError) as error:
        with service.query_context(deadline=time.monotonic() - 1):
            service.search_products("snap-test", {})
    assert error.value.code == "deadline_exceeded"
    cancel = threading.Event(); cancel.set()
    with pytest.raises(KnowledgeError) as error:
        with service.query_context(cancellation=cancel):
            service.search_products("snap-test", {})
    assert error.value.code == "cancelled"


def test_existing_agent_ports_validate_and_use_trusted_snapshot(tmp_path):
    from agent.contracts import CallContext
    from knowledge.agent_tools import build_knowledge_tools
    from pydantic import ValidationError
    root, _ = fixture_snapshot(tmp_path)
    registry = build_knowledge_tools(KnowledgeService(root))
    tool = registry.get("search_products")
    with pytest.raises(ValidationError):
        tool.arguments_model.model_validate({"filters": {}, "snapshot_id": "model-chosen"}, strict=True)
    async def run():
        context = CallContext("run", "snap-test", "operation", 5, asyncio.Event())
        arguments = tool.arguments_model.model_validate({"filters": {"category": "earbuds"}}, strict=True)
        results = await asyncio.gather(tool.executor.execute(arguments, context), tool.executor.execute(arguments, context))
        for result in results:
            parsed = tool.result_model.model_validate(result, strict=True)
            assert parsed.snapshot_id == "snap-test" and parsed.bytes <= tool.max_result_bytes
    try:
        asyncio.run(run())
    finally:
        tool.executor.close()


class BlockingKnowledge(KnowledgeService):
    """A controlled noncooperative operation to exercise worker admission."""
    def __init__(self, root):
        super().__init__(root)
        self.release_workers = threading.Event()
        self.started = 0
        self.started_lock = threading.Lock()

    def search_products(self, snapshot_id, **kwargs):
        with self.started_lock:
            self.started += 1
        self.release_workers.wait(timeout=3)
        return {"finished": True}


def test_timed_out_workers_keep_slots_until_actual_thread_completion(tmp_path, monkeypatch):
    from agent.contracts import CallContext, CapabilityError
    from knowledge.agent_tools import build_knowledge_tools
    service = BlockingKnowledge(tmp_path)
    registry = build_knowledge_tools(service)
    tool = registry.get("search_products")
    arguments = tool.arguments_model.model_validate({"filters": {}}, strict=True)
    original_submit = tool.executor.workers.pool.submit
    submitted = []
    def tracked_submit(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        submitted.append(future)
        return future
    monkeypatch.setattr(tool.executor.workers.pool, "submit", tracked_submit)
    async def run():
        def context(timeout):
            return CallContext("run", "snap-test", "op", timeout, asyncio.Event())
        first = await asyncio.gather(
            tool.executor.execute(arguments, context(0.03)),
            tool.executor.execute(arguments, context(0.03)), return_exceptions=True,
        )
        assert all(isinstance(error, CapabilityError) and error.issue.code == "deadline_exceeded" for error in first)
        assert service.started == len(submitted) == 2
        assert all(not future.done() for future in submitted)
        for _ in range(4):
            with pytest.raises(CapabilityError) as error:
                await tool.executor.execute(arguments, context(0.01))
            assert error.value.issue.code == "deadline_exceeded"
        assert len(submitted) == 2  # No hidden backlog after callers time out.
        assert tool.executor.workers.pool._work_queue.qsize() == 0
        assert tool.executor.workers.slots._value == 0
        service.release_workers.set()
        for _ in range(200):
            if all(future.done() for future in submitted) and tool.executor.workers.slots._value == 2:
                break
            await asyncio.sleep(0.001)
        assert tool.executor.workers.slots._value == 2
    try:
        asyncio.run(run())
    finally:
        service.release_workers.set()
        tool.executor.close()


def test_queued_worker_acquire_observes_cancellation_immediately(tmp_path):
    from agent.contracts import CallContext, CapabilityError
    from knowledge.agent_tools import build_knowledge_tools
    service = BlockingKnowledge(tmp_path)
    registry = build_knowledge_tools(service)
    tool = registry.get("search_products")
    arguments = tool.arguments_model.model_validate({"filters": {}}, strict=True)
    async def run():
        running = [asyncio.create_task(tool.executor.execute(
            arguments, CallContext("run", "snap-test", f"running-{i}", 2, asyncio.Event()))) for i in range(2)]
        for _ in range(200):
            if service.started == 2:
                break
            await asyncio.sleep(0.001)
        assert service.started == 2
        cancellation = asyncio.Event()
        queued = asyncio.create_task(tool.executor.execute(
            arguments, CallContext("run", "snap-test", "queued", 1, cancellation)))
        await asyncio.sleep(0.01)
        start = time.monotonic()
        cancellation.set()
        with pytest.raises(CapabilityError) as error:
            await asyncio.wait_for(queued, timeout=0.2)
        assert error.value.issue.code == "cancelled"
        assert time.monotonic() - start < 0.2 and service.started == 2
        assert tool.executor.workers.slots._value == 0
        service.release_workers.set()
        await asyncio.gather(*running)
    try:
        asyncio.run(run())
    finally:
        service.release_workers.set()
        tool.executor.close()


def test_missing_destination_and_quantity_cannot_establish_complete_cost(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE offers SET identity_status='verified' WHERE offer_id='offer-0'")
    connection.execute("UPDATE offer_observations SET shipping_scope='item',tax_minor=0,tax_status='known',price_includes_tax='excluded' WHERE offer_id='offer-0'")
    connection.execute("DELETE FROM current_offer_rows")
    connection.execute("INSERT INTO current_offer_rows SELECT * FROM _computed_current_offer_observations")
    connection.commit(); connection.close()
    fact = KnowledgeService(root).get_offer_facts("snap-test", ["offer-0"])["offers"][0]
    assert {"destination", "quantity"} <= set(fact["unknown_fields"])
    assert fact["complete_total_minor"] is None


def test_resolve_rejects_page_that_cannot_advance_one_unicode_character(tmp_path):
    import knowledge.service as module
    root, path = fixture_snapshot(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE evidence SET text=?,source_label='' WHERE evidence_id='long-review'", ("中" * 1401,))
    connection.commit(); connection.close()
    wide = KnowledgeService(root, max_result_bytes=65536)
    record = wide.resolve_evidence("snap-test", ["long-review"])["evidence"][0]
    record.update(text="", text_end=0, text_truncated=True)
    fingerprint = module._hash(["snap-test", ["long-review"], "resolve"])
    cursor = wide._cursor({"f": fingerprint, "i": 0, "o": 0})
    snapshot = wide._snapshot("snap-test")
    def empty_page():
        return wide._envelope(snapshot, evidence=[record], unknown_ids=[], next_cursor=cursor, truncated=True)
    budget = max(1024, size(empty_page()) + 30)
    # Leave exactly one byte: empty text would fit, a full UTF-8 character cannot.
    while size(empty_page()) < budget - 1:
        record["source_label"] += "x"
    while size(empty_page()) > budget - 1:
        record["source_label"] = record["source_label"][:-1]
    connection = sqlite3.connect(path)
    connection.execute("UPDATE evidence SET source_label=? WHERE evidence_id='long-review'", (record["source_label"],))
    connection.commit(); connection.close()
    with pytest.raises(KnowledgeError) as error:
        KnowledgeService(root, max_result_bytes=budget).resolve_evidence("snap-test", ["long-review"])
    assert error.value.code == "result_budget_exceeded"


def test_filter_discovery_uses_snapshot_schema_and_exposes_query_names_only(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    with sqlite3.connect(path) as connection:
        schema = json.loads(connection.execute(
            "SELECT schema_json FROM category_schemas WHERE category_id='earbuds'").fetchone()[0])
        schema["attributes"]["connectivity"] = {"type": "string", "aliases": ["wireless technology"]}
        connection.execute("UPDATE category_schemas SET schema_json=? WHERE category_id='earbuds'", (json.dumps(schema),))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    service = KnowledgeService(root)
    categories = service.describe_filters("snap-test")
    assert categories["categories"] == ["backpacks", "earbuds", "unseen-category"]
    assert categories["category"] is None and categories["attributes"] == []
    result = service.describe_filters("snap-test", "earbuds")
    attributes = {item["name"]: item for item in result["attributes"]}
    assert attributes["anc"] == {"name": "anc", "type": "boolean", "operators": ["eq"],
                                 "unit": None, "accepted_units": []}
    assert attributes["connectivity"]["operators"] == ["eq", "contains"]
    assert attributes["battery_life"]["unit"] == "h"
    assert attributes["battery_life"]["accepted_units"] == ["h", "min"]
    assert "aliases" not in json.dumps(result) and "wireless technology" not in json.dumps(result)
    assert result["categories"] == [] and result["category"] == "earbuds"
    # Exercise advertised numeric operators and units, rather than only checking metadata.
    for operator in attributes["battery_life"]["operators"]:
        page = service.search_products("snap-test", {"category": "earbuds", "variant_ids": ["v-0"],
            "hard_constraints": {"battery_life": {"op": operator, "value": 3, "unit": "min"}}})
        assert page["total_count"] == (1 if operator in ("gt", "gte") else 0)
        if page["candidates"]:
            assert page["candidates"][0]["hard_constraint_status"]["battery_life"] == "satisfied"
    assert result["bytes"] == size(result) <= service.max_result_bytes
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_discovery_and_validation_share_category_specific_attributes(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE category_schemas SET schema_json=? WHERE category_id='backpacks'",
                           (json.dumps({"attributes": {"capacity": {"type": "number", "unit": "L"}}}),))
    service = KnowledgeService(root)
    assert [a["name"] for a in service.describe_filters("snap-test", "backpacks")["attributes"]] == ["capacity"]
    with pytest.raises(KnowledgeError) as error:
        service.search_products("snap-test", {"category": "earbuds", "hard_constraints": {
            "capacity": {"op": "gte", "value": 20}}})
    assert error.value.recoverable and "battery_life" in error.value.public_message
    # A category-free query may still apply an attribute defined by another category.
    assert service.search_products("snap-test", {"hard_constraints": {
        "capacity": {"op": "gte", "value": 20}}})["total_count"] == 9


@pytest.mark.parametrize("filters,correction", [
    ({"category": "audio"}, "earbuds"),
    ({"category": "earbuds", "hard_constraints": {"type": {"op": "eq", "value": "wireless"}}}, "battery_life"),
    ({"category": "earbuds", "hard_constraints": {"anc": {"op": "gte", "value": True}}}, '"eq"'),
    ({"category": "earbuds", "hard_constraints": {"anc": {"op": ["eq"], "value": True}}}, '"eq"'),
    ({"category": "earbuds", "hard_constraints": {"battery_life": {"op": "contains", "value": 1}}}, '"gte"'),
    ({"category": "earbuds", "hard_constraints": {"anc": {"op": "eq", "value": "yes"}}}, "true or false"),
    ({"category": "earbuds", "hard_constraints": {"battery_life": {"op": "eq", "value": True}}}, "numeric"),
])
def test_correctable_query_errors_explain_how_to_repair(tmp_path, filters, correction):
    root, _ = fixture_snapshot(tmp_path)
    with pytest.raises(KnowledgeError) as error:
        KnowledgeService(root).search_products("snap-test", filters)
    assert error.value.code == "invalid_arguments" and error.value.recoverable
    assert correction in error.value.public_message
    assert len(error.value.public_message) <= 500


def test_trusted_host_and_snapshot_errors_remain_fatal(tmp_path):
    root, path = fixture_snapshot(tmp_path)
    with pytest.raises(KnowledgeError) as error:
        KnowledgeService(root, max_result_bytes=1)
    assert not error.value.recoverable
    service = KnowledgeService(root)
    for snapshot in ("missing", "../snap-test"):
        with pytest.raises(KnowledgeError) as error:
            service.describe_filters(snapshot)
        assert not error.value.recoverable
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value='wrong-id' WHERE key='snapshot_id'")
    with pytest.raises(KnowledgeError) as error:
        service.describe_filters("snap-test")
    assert error.value.code == "snapshot_mismatch" and not error.value.recoverable


def test_discovery_paginates_at_minimum_budget_and_error_feedback_stays_bounded(tmp_path):
    from agent.contracts import CallContext, CapabilityError
    from knowledge.agent_tools import DescribeFiltersResult, build_knowledge_tools
    root, path = fixture_snapshot(tmp_path)
    names = [f"property_{i:02d}_" + "长" * 24 for i in range(15)]
    categories = [f"category_{i:02d}_" + "新" * 24 for i in range(15)]
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE category_schemas SET schema_json=? WHERE category_id='earbuds'",
                           (json.dumps({"attributes": {name: {"type": "number", "unit": "h"} for name in names}}),))
        for category in categories:
            put(connection, "category_schemas", category_id=category, schema_json='{"attributes":{}}')
    service = KnowledgeService(root, max_result_bytes=1024)
    first = None
    for category, key, expected in ((None, "categories", sorted(categories + ["backpacks", "earbuds", "unseen-category"])),
                                    ("earbuds", "attributes", sorted(names))):
        cursor, collected, seen_cursors, pages = None, [], set(), 0
        while True:
            result = service.describe_filters("snap-test", category, cursor)
            if category == "earbuds" and first is None:
                first = result
            parsed = DescribeFiltersResult.model_validate(result, strict=True)
            assert parsed.bytes == size(result) <= 1024
            assert len(parsed.model_dump_json().encode()) <= 1024
            collected += result[key] if category is None else [a["name"] for a in result[key]]
            pages += 1
            cursor = result["next_cursor"]
            if not cursor:
                assert not result["truncated"]
                break
            assert result["truncated"] and cursor not in seen_cursors
            seen_cursors.add(cursor)
        assert collected == expected and pages > 1
    with pytest.raises(KnowledgeError) as error:
        service.describe_filters("snap-test", "backpacks", first["next_cursor"])
    assert error.value.recoverable and error.value.code == "invalid_arguments"
    registry = build_knowledge_tools(service)
    tool = registry.get("search_products")
    async def query():
        arguments = tool.arguments_model.model_validate({"filters": {"category": "earbuds", "hard_constraints": {
            "bad_" * 1000: {"op": "eq", "value": 1}}}}, strict=True)
        with pytest.raises(CapabilityError) as error:
            await tool.executor.execute(arguments, CallContext("run", "snap-test", "op", 5, asyncio.Event()))
        assert error.value.recoverable and len(error.value.issue.message) <= 500
        assert "describe_filters" in error.value.issue.message
    try:
        asyncio.run(query())
    finally:
        tool.executor.close()


def test_discovery_adapter_uses_trusted_snapshot_and_is_visible_to_model(tmp_path):
    from agent import AgentRequest, RunPolicy
    from agent.contracts import CallContext
    from agent.prompts import build_model_request
    from agent.state import initial_state
    from inference.model_port import to_chat_messages
    from knowledge.agent_tools import build_knowledge_tools
    from pydantic import ValidationError
    root, path = fixture_snapshot(tmp_path)
    second = root / "snap-second"
    second.mkdir()
    other_db = second / "catalog.sqlite"
    other_db.write_bytes(path.read_bytes())
    with sqlite3.connect(other_db) as connection:
        connection.execute("UPDATE metadata SET value='snap-second' WHERE key='snapshot_id'")
        connection.execute("DELETE FROM category_schemas")
        put(connection, "category_schemas", category_id="only-second", schema_json='{"attributes":{}}')
    registry = build_knowledge_tools(KnowledgeService(root))
    tool = registry.get("describe_filters")
    with pytest.raises(ValidationError):
        tool.arguments_model.model_validate({"snapshot_id": "snap-second"}, strict=True)
    policy = RunPolicy()
    state = initial_state(AgentRequest(run_id="r", conversation_id="c", snapshot_id="snap-test",
                                      message="Recommend earbuds"), policy)
    request = build_model_request(state, registry, policy)
    description = next(t for t in request.tools if t.name == "describe_filters")
    assert {"category", "cursor"} == set(description.parameters["properties"])
    assert "describe_filters" in to_chat_messages(request)[0]["content"]
    async def query():
        results = await asyncio.gather(*[
            tool.executor.execute(tool.arguments_model(), CallContext("run", snapshot, "op", 5, asyncio.Event()))
            for snapshot in ("snap-test", "snap-second")])
        for result in results:
            tool.result_model.model_validate(result, strict=True)
        assert results[0]["categories"] == ["backpacks", "earbuds", "unseen-category"]
        assert results[1]["categories"] == ["only-second"]
    try:
        asyncio.run(query())
    finally:
        tool.executor.close()


@pytest.mark.parametrize("category,attribute,value,bad_filters", [
    ("earbuds", "anc", True, {"category": "audio"}),
    ("keyboards", "switch_type", "mechanical", {"category": "keyboards", "hard_constraints": {
        "type": {"op": "eq", "value": "mechanical"}}}),
])
def test_issue4_real_snapshot_runtime_repairs_arguments_without_category_hints(
        tmp_path, category, attribute, value, bad_filters):
    from agent import AgentDependencies, AgentRequest, AgentRuntime, ModelReply
    from knowledge.agent_tools import build_knowledge_tools
    from knowledge.bootstrap import install_snapshot
    from knowledge.settings import load_settings
    settings = load_settings(data_root=tmp_path, snapshots_root=tmp_path / "snapshots")
    install_snapshot(settings)
    registry = build_knowledge_tools(KnowledgeService(settings.snapshots_root, **settings.service))
    class RepairModel:
        def __init__(self):
            self.requests = []

        async def generate(self, request, context):
            step = len(self.requests)
            self.requests.append(request)
            if step:
                observation = json.loads(request.messages[-1].content)["observation"]
            if step == 0:
                action = {"type": "call_tool", "tool_name": "search_products", "arguments": {"filters": bad_filters}}
            elif step == 1:
                assert observation["status"] == "error" and observation["code"] == "invalid_arguments"
                assert "describe_filters" in observation["message"]
                action = {"type": "call_tool", "tool_name": "describe_filters", "arguments": {}}
            elif step == 2:
                assert observation["status"] == "ok" and category in observation["data"]["categories"]
                action = {"type": "call_tool", "tool_name": "describe_filters", "arguments": {"category": category}}
            elif step == 3:
                assert observation["status"] == "ok"
                descriptor = next(a for a in observation["data"]["attributes"] if a["name"] == attribute)
                assert "eq" in descriptor["operators"]
                action = {"type": "call_tool", "tool_name": "search_products", "arguments": {"limit": 1,
                    "filters": {"category": category, "hard_constraints": {
                        descriptor["name"]: {"op": "eq", "value": value}}}}}
            else:
                assert step == 4 and observation["status"] == "ok" and observation["tool"] == "search_products"
                action = {"type": "ask_user", "question": "What is your delivery destination?",
                          "required_fields": ["destination"]}
            return ModelReply(content=json.dumps(action))
    model = RepairModel()
    runtime = AgentRuntime(AgentDependencies(model=model, tools=registry))
    request = AgentRequest(run_id="repair", conversation_id="repair", snapshot_id=settings.snapshot_id,
                           message="Recommend something suitable for my commute, under 100 USD.",
                           constraints={"currency": "USD", "purchase_cap": "100.00"})
    try:
        result = asyncio.run(runtime.run(request))
        assert result.status == "waiting_user" and result.verified_plan is None
        assert result.usage.model_calls == 5 and result.usage.tool_calls == 4
        assert any(i.code == "invalid_arguments" for i in result.issues)
        assert all("available_categories" not in m.content for r in model.requests for m in r.messages)
    finally:
        registry.get("describe_filters").executor.close()


def test_correctable_errors_still_exhaust_the_existing_call_budget(tmp_path):
    from agent import AgentDependencies, AgentRequest, AgentRuntime, ModelReply, RunPolicy
    from knowledge.agent_tools import build_knowledge_tools
    root, _ = fixture_snapshot(tmp_path)
    registry = build_knowledge_tools(KnowledgeService(root))
    class RepeatingModel:
        async def generate(self, request, context):
            return ModelReply(content=json.dumps({"type": "call_tool", "tool_name": "search_products",
                                                "arguments": {"filters": {"category": "audio"}}}))
    runtime = AgentRuntime(AgentDependencies(model=RepeatingModel(), tools=registry), RunPolicy(max_tool_calls=2))
    try:
        result = asyncio.run(runtime.run(AgentRequest(run_id="r", conversation_id="c", snapshot_id="snap-test",
                                                    message="Recommend earbuds")))
        assert result.status == "incomplete" and result.stop_reason == "tool_call_limit"
        assert result.usage.tool_calls == 2 and result.usage.model_calls == 3
    finally:
        registry.get("describe_filters").executor.close()
