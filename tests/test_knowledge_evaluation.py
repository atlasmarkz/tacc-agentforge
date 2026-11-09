"""Offline structural-label, retrieval-metric and honest-benchmark regressions."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge.benchmark import benchmark
from knowledge.evaluation import evaluate, prepare_development_labels, retrieval_metrics
from knowledge.store import export_snapshot, ingest_runs, verify_snapshot


REPOSITORY = Path(__file__).resolve().parents[1]


class MetricTests(unittest.TestCase):
    def test_recall_and_first_relevant_rank(self):
        result = retrieval_metrics(["x", "a", "b"], ["a", "b", "c"])
        self.assertAlmostEqual(result["recall_at_5"], 2 / 3)
        self.assertEqual(result["mrr_at_5"], 0.5)

    def test_group_equivalent_ids_do_not_inflate_denominator_or_rank(self):
        result = retrieval_metrics(["x", "a2", "a", "b"], ["a", "a2", "b"],
                                   {"a": "group-a", "a2": "group-a"})
        self.assertEqual(result["recall_at_5"], 1)
        self.assertEqual(result["mrr_at_5"], 0.5)
        self.assertEqual(result["relevant_groups"], 2)
        self.assertEqual(result["retrieved_groups"], 3)

    def test_no_hit_and_top_five_boundary(self):
        result = retrieval_metrics(["x1", "x2", "x3", "x4", "x5", "gold"], ["gold"])
        self.assertEqual(result["recall_at_5"], 0)
        self.assertEqual(result["mrr_at_5"], 0)

    def test_empty_gold_is_undefined(self):
        result = retrieval_metrics(["x"], [])
        self.assertIsNone(result["recall_at_5"])
        self.assertIsNone(result["mrr_at_5"])

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([], [], k=0)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        scratch = REPOSITORY / "output"
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix=".kb-evaluation-test-", dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "raw").mkdir(parents=True)
        self.config = self.root / "queries.json"
        categories = ("earbuds", "chargers", "keyboards", "backpacks")
        self.config.write_text(json.dumps({"queries": [
            {"q": "fixture " + category, "category": category} for category in categories
        ]}))
        self.index = []
        for category in categories:
            self._record(category + "-search", "google_shopping", {"q": "fixture " + category}, {
                "shopping_results": [
                    {"product_id": category + str(i), "immersive_product_page_token": category + "-token-" + str(i),
                     "title": f"Fixture {category} {i}", "price": "USD 20.00", "extracted_price": 20,
                     "source": "Fixture Shop", "product_link": f"https://example.com/{category}/{i}"}
                    for i in range(2)
                ]
            })
            for i in range(2):
                attributes = {
                    "earbuds": [("Noise Canceling", "Yes" if i == 0 else "No"),
                                ("Max Battery Life", f"{10 + i} hours"), ("Wireless Type", "Bluetooth")],
                    "chargers": [("Watts", f"{65 + i} W"), ("Capacity", f"{20000 + i} mAh"), ("Ports", "2")],
                    "keyboards": [("Wireless", "Yes" if i == 0 else "No"),
                                  ("Language / Layout", "English"), ("Switch Type", "Mechanical")],
                    "backpacks": [("Capacity", f"{20 + i} L"), ("Waterproof", "Yes"), ("Material", "Nylon")],
                }[category]
                self._record(category + "-detail-" + str(i), "google_product",
                             {"page_token": category + "-token-" + str(i)}, {
                    "product_results": {
                        "title": f"Fixture {category} {i}", "price_insights": {"currency": "USD"},
                        "product_attributes": [{"name": name, "value": value} for name, value in attributes],
                        "about_the_product": {"description": f"Everyday fixture {category} {i}."},
                        "stores": [{"name": "Fixture Shop", "title": f"Fixture {category} {i}",
                                    "link": f"https://example.com/{category}/{i}", "price": "USD 20.00",
                                    "extracted_price": 20, "shipping": "Free", "additional_costs": "+tax"}],
                        "user_reviews": [{"text": f"The fixture {category} {i} worked for my daily use.", "rating": 4}],
                    }
                })
        (self.run / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.index))
        self.db = self.root / "knowledge" / "catalog.sqlite"
        ingest_runs([self.run], self.db, self.config, REPOSITORY / "configs/category_schemas.json")
        exported = export_snapshot(self.db, self.root / "snapshots")
        self.snapshot_id = exported["snapshot_id"]
        self.snapshot_dir = Path(exported["path"])
        self.snapshots_root = self.snapshot_dir.parent
        self.labels_path = self.root / "development-labels.json"

    def _record(self, call_id, engine, params, body):
        payload = json.dumps(body, sort_keys=True).encode()
        relative = "raw/" + call_id + ".json"
        (self.run / relative).write_bytes(payload)
        self.index.append({"call_id": call_id, "engine": engine, "status": "ok",
                           "params": {"gl": "us", "hl": "en", **params}, "raw_path": relative,
                           "sha256": sha256(payload).hexdigest(), "requested_at": "2026-11-04T12:00:00Z"})

    def test_prepare_labels_uses_sql_not_retriever_and_balances_categories(self):
        before = (self.snapshot_dir / "catalog.sqlite").read_bytes()
        with patch("knowledge.evaluation.KnowledgeService", side_effect=AssertionError("must not retrieve")):
            result = prepare_development_labels(self.snapshot_dir, self.labels_path)
        self.assertEqual(result["case_count"], 10)
        self.assertEqual(len(result["category_counts"]), 4)
        self.assertLessEqual(max(result["category_counts"].values()) - min(result["category_counts"].values()), 1)
        labels = json.loads(self.labels_path.read_text())
        self.assertEqual(labels["label_kind"], "programmatic_structural_not_human_relevance")
        self.assertFalse(labels["generation"]["retriever_queried"])
        self.assertTrue(all(len(case["filters"]["variant_ids"]) == 1 for case in labels["cases"]))
        self.assertTrue(all(len(case["filters"]["hard_constraints"]) == 1 for case in labels["cases"]))
        self.assertTrue(all(case["relevant_evidence_ids"] for case in labels["cases"]))
        self.assertEqual(before, (self.snapshot_dir / "catalog.sqlite").read_bytes())

    def test_frozen_labels_are_not_overwritten_or_written_inside_snapshot(self):
        prepare_development_labels(self.snapshot_dir, self.labels_path)
        before = self.labels_path.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_development_labels(self.snapshot_dir, self.labels_path)
        self.assertEqual(before, self.labels_path.read_bytes())
        with self.assertRaises(ValueError):
            prepare_development_labels(self.snapshot_dir, self.snapshot_dir / "labels.json")

    def test_compare_modes_keeps_candidate_prices_and_snapshot_fixed(self):
        prepare_development_labels(self.snapshot_dir, self.labels_path)
        before = (self.snapshot_dir / "catalog.sqlite").read_bytes()
        result = evaluate(self.snapshots_root, self.snapshot_id, self.labels_path)
        self.assertEqual(result["case_count"], 10)
        self.assertTrue(result["same_candidate_and_price_scope"])
        self.assertTrue(result["source_snapshot_unchanged"])
        for case in result["cases"]:
            self.assertEqual(len(case["candidate_variant_ids"]), 1)
            self.assertTrue(case["same_candidate_and_price_scope"])
            for scores in case["modes"].values():
                self.assertGreaterEqual(scores["recall_at_5"], 0)
                self.assertLessEqual(scores["recall_at_5"], 1)
                self.assertEqual(scores["id_validity"], 1)
                self.assertEqual(scores["citation_provenance_coverage"], 1)
                self.assertLessEqual(scores["serialized_response_bytes"], 8192)
        self.assertEqual(before, (self.snapshot_dir / "catalog.sqlite").read_bytes())
        self.assertTrue(verify_snapshot(self.snapshot_dir)["valid"])

    def test_unknown_gold_id_is_rejected(self):
        prepare_development_labels(self.snapshot_dir, self.labels_path)
        labels = json.loads(self.labels_path.read_text())
        labels["cases"][0]["relevant_evidence_ids"] = ["evidence_missing"]
        changed = self.root / "invalid-labels.json"
        changed.write_text(json.dumps(labels))
        with self.assertRaisesRegex(ValueError, "known, nonempty"):
            evaluate(self.snapshots_root, self.snapshot_id, changed)

    def test_benchmark_scales_only_disposable_evidence_and_serializes_dtos(self):
        before = (self.snapshot_dir / "catalog.sqlite").read_bytes()
        result = benchmark(self.snapshots_root, self.snapshot_id, iterations=20, scale=2)
        self.assertTrue(result["source_snapshot_unchanged"])
        self.assertEqual(result["synthetic"]["counts"]["evidence"], result["actual"]["counts"]["evidence"] * 2)
        self.assertEqual(result["synthetic"]["counts"]["variants"], result["actual"]["counts"]["variants"])
        self.assertTrue(result["synthetic"]["synthetic_snapshot_verified"])
        for corpus in (result["actual"], result["synthetic"]):
            for profile in corpus["profiles"].values():
                for workload in profile.values():
                    for timing in workload.values():
                        self.assertEqual(timing["calls"], 20)
                        self.assertGreaterEqual(timing["p50_ms"], 0)
                        self.assertLessEqual(timing["max_response_bytes"], 8192)
                        self.assertEqual(timing["declared_byte_count_mismatches"], 0)
            self.assertIn("resolve_provenance_probe", corpus["sql_without_service_result_cache"]["probes"])
            self.assertTrue(corpus["gates"]["all_measured_dtos_at_most_8192_bytes"])
        self.assertEqual(before, (self.snapshot_dir / "catalog.sqlite").read_bytes())
        self.assertTrue(verify_snapshot(self.snapshot_dir)["valid"])


if __name__ == "__main__":
    unittest.main()
