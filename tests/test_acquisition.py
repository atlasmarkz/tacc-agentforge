"""Offline fake-HTTP checks for incremental acquisition; no sockets or credentials."""
from functools import partial
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from knowledge import acquisition as ac
from knowledge.serpapi import SerpApiClient


KEY = "FICTIONAL-ACQUISITION-KEY"


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data-root"
        self.runs = self.data / "data" / "serpapi"
        self.old = self.runs / "run-old"
        (self.old / "raw").mkdir(parents=True)
        self.session = self.runs / "run-new"
        self.config = self.root / "queries.json"
        self.categories = ["earbuds", "chargers", "keyboards", "backpacks"]
        self.queries = [{"category": cat, "q": cat + " query"} for cat in self.categories]
        self.config.write_text(json.dumps({"locale": {"gl": "us", "hl": "en"}, "queries": self.queries}))
        self.records = []
        self.fixture(15)

    def record(self, call_id, engine, params, document, status="ok"):
        body = json.dumps(document).encode()
        raw = "raw/" + call_id + ".json"
        (self.old / raw).write_bytes(body)
        self.records.append({"call_id": call_id, "engine": engine, "params": params,
            "requested_at": "2026-11-01T00:00:00+00:00", "processed_at": "",
            "status": status, "raw_path": raw, "sha256": hashlib.sha256(body).hexdigest()})

    def fixture(self, per_category):
        self.records = []
        for n, query in enumerate(self.queries):
            rows = [{"product_id": f"{n}-{i}", "title": f"Product {n}-{i}", "position": i + 1,
                     "immersive_product_page_token": f"old-token-{n}-{i}"} for i in range(per_category)]
            self.record(f"search-{n}", "google_shopping", {"q": query["q"]}, {"shopping_results": rows})
            self.record(f"detail-{n}", "google_product", {"page_token": f"old-token-{n}-0"},
                        {"product_results": {"title": f"Product {n}-0"}})
        (self.old / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def invoke(self, transport, cap=50, target=1):
        # Reuse the production HTTP handling and retry logic, replacing only I/O and sleeps.
        with patch.object(SerpApiClient, "_http_get", transport), patch.object(
                ac, "SerpApiClient", partial(SerpApiClient, sleep=lambda _: None)):
            return ac.collect_incremental(self.data, self.config, self.session, KEY,
                                          max_attempts=cap, target_per_category=target)

    def test_plan_is_read_only_round_robin_and_skips_global_successes(self):
        before = sorted(str(p.relative_to(self.data)) for p in self.data.rglob("*"))
        plan = ac.plan_collection(self.data, self.config, target_per_category=2)
        self.assertEqual(plan["already_completed_details"], 4)
        self.assertEqual(plan["planned_new_details"], 8)
        self.assertEqual([t["category"] for t in plan["targets"]], self.categories * 2)
        self.assertEqual([t["product_id"] for t in plan["targets"][:4]], [f"{i}-1" for i in range(4)])
        self.assertEqual(before, sorted(str(p.relative_to(self.data)) for p in self.data.rglob("*")))

    def test_persistent_fifty_attempt_cap_survives_restart(self):
        calls = []
        def transport(client, params):
            calls.append(dict(params))
            return 200, json.dumps({"product_results": {"title": params.get("page_token", "detail")}})
        first = self.invoke(transport, cap=50, target=13)
        self.assertEqual(first["attempts_used"], 50)
        self.assertEqual(first["completed_new_details"], 50)
        self.assertEqual(first["stopped"], "budget_exhausted")
        second = self.invoke(transport, cap=50, target=13)
        self.assertEqual(len(calls), 50)
        self.assertEqual(second["attempts_used"], 50)
        self.assertEqual(second["attempts_this_invocation"], 0)
        self.assertEqual(second["planned_new_details"], 52)
        with self.assertRaisesRegex(ValueError, "original"):
            self.invoke(transport, cap=51, target=13)

    def test_queries_rotate_within_category_and_keep_global_deduplication(self):
        extra = {"category": "earbuds", "q": "noise cancelling earbuds"}
        self.queries.insert(1, extra)
        self.config.write_text(json.dumps({"locale": {"gl": "us", "hl": "en"}, "queries": self.queries}))
        # First row deliberately duplicates the first query's first new product.
        self.record("extra-search", "google_shopping", {"q": extra["q"]}, {"shopping_results": [
            {"product_id": "0-1", "position": 1, "immersive_product_page_token": "duplicate-token"},
            {"product_id": "extra-1", "position": 2, "immersive_product_page_token": "extra-token-1"},
            {"product_id": "extra-2", "position": 3, "immersive_product_page_token": "extra-token-2"}]})
        (self.old / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))
        plan = ac.plan_collection(self.data, self.config, target_per_category=4)
        earbuds = [t for t in plan["targets"] if t["category"] == "earbuds"]
        self.assertEqual([t["query"] for t in earbuds], ["earbuds query", extra["q"]] * 2)
        self.assertEqual([t["product_id"] for t in earbuds], ["0-1", "extra-1", "0-2", "extra-2"])
        self.assertEqual(len({t["product_id"] for t in plan["targets"]}), len(plan["targets"]))
        self.assertEqual([t["category"] for t in plan["targets"]], self.categories * 4)

    def test_auth_json_and_non_json_are_logged_once_and_stop(self):
        for status, body in [(401, json.dumps({"error": "Invalid " + KEY})),
                             (403, "<html>access denied " + KEY + "</html>")]:
            with self.subTest(status=status):
                self.session = self.runs / f"run-auth-{status}"
                calls = []
                def transport(client, params):
                    calls.append(params)
                    return status, body
                result = self.invoke(transport)
                self.assertEqual(result["stopped"], "authentication_failed")
                self.assertEqual(result["attempts_used"], 1)
                self.assertEqual(len(calls), 1)
                rows = [json.loads(x) for x in (self.session / "index.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "error")
                self.assertEqual(rows[0]["http_status"], status)
                with sqlite3.connect(self.session / "acquisition.sqlite") as db:
                    self.assertEqual(db.execute("SELECT status FROM attempts").fetchone()[0], "error")
                for path in self.session.rglob("*"):
                    if path.is_file():
                        self.assertNotIn(KEY.encode(), path.read_bytes(), str(path))

    def test_retry_attempts_count_and_failed_target_can_resume(self):
        calls = []
        def failing(client, params):
            calls.append(params)
            return 500, "temporary failure"
        result = self.invoke(failing, cap=5, target=1)
        self.assertEqual(result["attempts_used"], 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["completed_new_details"], 0)
        self.assertEqual(ac.plan_collection(self.data, self.config, 1)["planned_new_details"], 4)
        second = self.invoke(failing, cap=5, target=1)
        self.assertEqual(second["attempts_this_invocation"], 0)

    def test_successful_resume_does_not_add_more_targets_or_calls(self):
        calls = []
        def transport(client, params):
            calls.append(params)
            return 200, json.dumps({"product_results": {"title": "detail"}})
        first = self.invoke(transport)
        second = self.invoke(transport)
        self.assertEqual(first["completed_by_category"], dict.fromkeys(self.categories, 1))
        self.assertEqual(second["completed_new_details"], 4)
        self.assertEqual(second["attempts_this_invocation"], 0)
        self.assertEqual(len(calls), 4)
        next_plan = ac.plan_collection(self.data, self.config, 1)
        self.assertEqual(next_plan["already_completed_details"], 8)
        self.assertEqual(next_plan["targets"][0]["product_id"], "0-2")
        rows = [json.loads(x) for x in (self.session / "index.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["category"] and r["product_id"] and r["parent_call_id"] and r["target_id"] for r in rows))
        self.assertTrue(all("category" not in r["params"] and "target_id" not in r["params"] for r in rows))

    def test_refresh_once_per_query_and_rematch_only_original_product_id(self):
        calls = []
        def transport(client, params):
            calls.append(dict(params))
            if params["engine"] == "google_shopping":
                return 200, json.dumps({"shopping_results": [
                    {"product_id": "wrong-rank-one", "immersive_product_page_token": "wrong-token"},
                    {"product_id": "0-1", "immersive_product_page_token": "new-token-0-1"},
                    {"product_id": "0-2", "immersive_product_page_token": "new-token-0-2"}]})
            if params.get("page_token") in {"old-token-0-1", "old-token-0-2"}:
                return 200, json.dumps({"error": "Invalid page token"})
            return 200, json.dumps({"product_results": {"title": "detail"}})
        result = self.invoke(transport, target=2)
        self.assertEqual(result["completed_new_details"], 8)
        self.assertEqual(sum(p["engine"] == "google_shopping" for p in calls), 1)
        used_tokens = [p.get("page_token") for p in calls]
        self.assertIn("new-token-0-1", used_tokens)
        self.assertIn("new-token-0-2", used_tokens)
        self.assertNotIn("wrong-token", used_tokens)

    def test_missing_original_id_on_refresh_is_never_replaced_by_rank(self):
        calls = []
        def transport(client, params):
            calls.append(dict(params))
            if params["engine"] == "google_shopping":
                return 200, json.dumps({"shopping_results": [{"product_id": "another", "immersive_product_page_token": "wrong-token"}]})
            if params.get("page_token") == "old-token-0-1":
                return 200, json.dumps({"error": "Google Product hasn't returned any results for this query."})
            return 200, json.dumps({"product_results": {"title": "detail"}})
        result = self.invoke(transport)
        self.assertEqual(result["completed_new_details"], 3)
        self.assertNotIn("wrong-token", [p.get("page_token") for p in calls])

    def test_task_lock_rejects_second_collector(self):
        path = self.runs / ".acquisition.lock"
        with ac._task_lock(path):
            with self.assertRaisesRegex(RuntimeError, "Another collector"):
                with ac._task_lock(path):
                    self.fail("Should not acquire a held task lock")


if __name__ == "__main__":
    unittest.main()
