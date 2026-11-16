"""Catalog expansion for the cloud deployment (issue #25): plan from a working catalog when the original run
directories are gone, per-category targets, and discovery of new shopping queries. Fake HTTP only."""
from functools import partial
import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from knowledge import acquisition as ac
from knowledge.serpapi import SerpApiClient
from tests.test_acquisition import AcquisitionTests

KEY = "FICTIONAL-ACQUISITION-KEY"
REPO_SCHEMAS = ac.Path(__file__).resolve().parents[1] / "configs/category_schemas.json"


@pytest.fixture()
def world(tmp_path):
    """The fixture of tests/test_acquisition.py: 4 categories, 15 discovered and 1 detailed product each."""
    case = AcquisitionTests("test_plan_is_read_only_round_robin_and_skips_global_successes")
    case.setUp()
    yield case
    case.doCleanups()


def fake_search(results_per_query=3):
    calls = []

    def transport(client, params):
        calls.append(dict(params))
        if params["engine"] == "google_shopping":
            rows = [{"product_id": f"{params['q']}-{i}", "title": f"{params['q']} {i}", "position": i + 1,
                     "immersive_product_page_token": f"tok-{params['q']}-{i}"} for i in range(results_per_query)]
            return 200, json.dumps({"shopping_results": rows})
        return 200, json.dumps({"product_results": {"title": params.get("page_token")}})
    return transport, calls


def run_with(transport, fn, *args, **kwargs):
    with patch.object(SerpApiClient, "_http_get", transport), patch.object(
            ac, "SerpApiClient", partial(SerpApiClient, sleep=lambda _: None)):
        return fn(*args, **kwargs)


def test_per_category_targets(world):
    plan = ac.plan_collection(world.data, world.config, {"earbuds": 3, "keyboards": 1})
    assert plan["category_counts"] == {"earbuds": 3, "keyboards": 1}
    assert plan["planned_new_details"] == 4 and plan["shortfalls"] == {}
    plan = ac.plan_collection(world.data, world.config, {"earbuds": 20})
    assert plan["category_counts"] == {"earbuds": 14} and plan["shortfalls"] == {"earbuds": 6}
    for bad in ({"tents": 2}, {"earbuds": -1}, {"earbuds": 0}, 0, True):
        with pytest.raises(ValueError):
            ac.plan_collection(world.data, world.config, bad)


def test_plan_from_a_working_catalog_matches_the_runs_it_came_from(world):
    from knowledge.store import ingest_runs
    db = world.root / "catalog/catalog.sqlite"
    ingest_runs([world.old], db, world.config, REPO_SCHEMAS)
    from_runs = ac.plan_collection(world.data, world.config, 2)
    shutil.rmtree(world.old)                                   # the original run directories are gone
    assert ac.plan_collection(world.data, world.config, 2)["planned_new_details"] == 0
    from_catalog = ac.plan_collection(world.data, world.config, 2, db_path=db)
    assert from_catalog["already_completed_details"] == from_runs["already_completed_details"] == 4
    assert [t["product_id"] for t in from_catalog["targets"]] == [t["product_id"] for t in from_runs["targets"]]
    assert all(t["page_token"] for t in from_catalog["targets"])        # detail lookups can use the tokens
    raw = next((db.parent / "raw").iterdir())
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        ac.plan_collection(world.data, world.config, 2, db_path=db)


def test_discovery_runs_only_new_queries_and_planning_picks_them_up(world):
    new = {"category": "mice", "q": "wireless mouse"}
    world.queries.append(new)
    world.config.write_text(json.dumps({"locale": {"gl": "us", "hl": "en"}, "queries": world.queries}))
    transport, calls = fake_search()
    summary = run_with(transport, ac.discover_queries, world.data, world.config,
                       world.runs / "run-discover-1", KEY, max_attempts=5)
    assert [c["q"] for c in calls] == ["wireless mouse"]                 # the four old queries already have results
    assert summary["stopped"] == "finished" and summary["by_category"] == {"mice": 1}
    assert summary["results_per_query"] == {"wireless mouse": 3}
    plan = ac.plan_collection(world.data, world.config, {"mice": 2})
    assert [t["product_id"] for t in plan["targets"]] == ["wireless mouse-0", "wireless mouse-1"]
    assert plan["targets"][0]["page_token"] == "tok-wireless mouse-0"
    with pytest.raises(ValueError, match="new session"):
        run_with(transport, ac.discover_queries, world.data, world.config, world.runs / "run-discover-1", KEY)


def test_discovery_refresh_and_attempt_cap(world):
    transport, calls = fake_search()
    summary = run_with(transport, ac.discover_queries, world.data, world.config, world.runs / "run-refresh",
                       KEY, max_attempts=2, refresh_existing=True)
    assert len(calls) == 2 and summary["stopped"] == "budget_exhausted" and summary["queries_run"] == 2
    assert KEY not in (world.runs / "run-refresh/index.jsonl").read_text()
    with pytest.raises(ValueError, match="run-"):
        run_with(transport, ac.discover_queries, world.data, world.config, world.root / "elsewhere", KEY)


def test_cli_parses_category_targets():
    from knowledge.cli import _targets
    assert _targets(SimpleNamespace(category_targets="earbuds=14, mice=43", target_per_category=10)) == \
        {"earbuds": 14, "mice": 43}
    assert _targets(SimpleNamespace(category_targets=None, target_per_category=7)) == 7
    with pytest.raises(ValueError):
        _targets(SimpleNamespace(category_targets="earbuds:14", target_per_category=1))
