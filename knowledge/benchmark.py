"""Local, reproducible KB timings with explicit cache and serialization boundaries.

Synthetic scaling repeats evidence in a disposable database; it does not create
new products, independently sourced facts, or additional human relevance labels.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time

from .evaluation import _open, _snapshot_path, _tree_fingerprint
from .service import KnowledgeService
from .store import export_snapshot, verify_snapshot


def _wire(result):
    # Matches the compact, Unicode-preserving adapter encoding boundary.
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _percentile(values, quantile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)] if ordered else None


def _sample(operation):
    started = time.perf_counter_ns()
    response = operation()
    serialized = _wire(response)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    return {"latency_ms": elapsed, "response_bytes": len(serialized),
            "declared_bytes_match": response.get("bytes") == len(serialized),
            "truncated": bool(response.get("truncated")),
            "has_next_cursor": bool(response.get("next_cursor"))}


def _summary(samples):
    latencies = [sample["latency_ms"] for sample in samples]
    sizes = [sample["response_bytes"] for sample in samples]
    return {"calls": len(samples), "p50_ms": statistics.median(latencies),
            "p95_ms": _percentile(latencies, 0.95), "max_ms": max(latencies),
            "min_response_bytes": min(sizes), "median_response_bytes": statistics.median(sizes),
            "max_response_bytes": max(sizes),
            "declared_byte_count_mismatches": sum(not x["declared_bytes_match"] for x in samples),
            "truncated_responses": sum(x["truncated"] for x in samples),
            "responses_with_next_cursor": sum(x["has_next_cursor"] for x in samples)}


def _source_workload(snapshot_dir):
    conn = _open(snapshot_dir / "catalog.sqlite")
    try:
        counts = {name: conn.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                  for name in ("variants", "listings", "offers", "offer_observations", "attributes", "reviews", "evidence")}
        row = conn.execute(
            "SELECT a.name,a.value FROM current_attributes a JOIN evidence e ON e.evidence_id=a.evidence_id "
            "WHERE a.status='known' ORDER BY e.evidence_id LIMIT 1").fetchone()
        if row:
            query = str(row["name"]) + " " + str(row["value"])
        else:
            row = conn.execute("SELECT title FROM evidence ORDER BY evidence_id LIMIT 1").fetchone()
            query = row[0] if row else ""
        tokens = list(dict.fromkeys(re.findall(r"\w+(?:[-']\w+)*", query)))[:12]
        if not tokens:
            raise ValueError("Snapshot needs searchable evidence for benchmarking")
        query = " ".join(tokens)
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        evidence_ids = [r[0] for r in conn.execute("SELECT evidence_id FROM evidence ORDER BY evidence_id LIMIT 3")]
        offer_ids = [r[0] for r in conn.execute("SELECT offer_id FROM offers ORDER BY offer_id LIMIT 3")]
        category = conn.execute("SELECT category FROM variants GROUP BY category ORDER BY COUNT(*) DESC,category LIMIT 1").fetchone()
        return {"counts": counts, "query": query, "match": match, "evidence_ids": evidence_ids,
                "offer_ids": offer_ids, "category": category[0] if category else None}
    finally:
        conn.close()


def _operations(service, snapshot_id, workload):
    # Complete structured scope is created once; returned DTO can be one page.
    started = time.perf_counter_ns()
    setup = service.search_products(snapshot_id, {}, limit=1)
    setup_bytes = len(_wire(setup))
    setup_ms = (time.perf_counter_ns() - started) / 1_000_000
    candidate_set_id = setup["candidate_set_id"]
    scope = service.describe_candidate_set(snapshot_id, candidate_set_id)
    operations = {
        "search_products": lambda: service.search_products(snapshot_id, {}, limit=5),
        "retrieve_evidence": lambda: service.retrieve_evidence(
            snapshot_id, workload["query"], candidate_set_id, k=5, mode="filtered"),
        "resolve_evidence": lambda: service.resolve_evidence(snapshot_id, workload["evidence_ids"]),
    }
    if workload["offer_ids"]:
        operations["get_offer_facts"] = lambda: service.get_offer_facts(snapshot_id, workload["offer_ids"])
    return operations, {"candidate_setup_ms": setup_ms, "candidate_setup_response_bytes": setup_bytes,
                        "candidate_count": len(scope["variant_ids"]),
                        "selected_price_records": len(scope["selected_price_record_ids"]),
                        "candidate_set_id": candidate_set_id,
                        "excluded_from_retrieve_evidence_timings": True}


def _concurrent(operation, iterations):
    # Two closed-loop workers, rather than a queue of 100 simultaneous requests.
    def worker(number):
        return [_sample(operation) for _ in range(number)]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="kb-benchmark") as executor:
        first = executor.submit(worker, (iterations + 1) // 2)
        second = executor.submit(worker, iterations // 2)
        return first.result() + second.result()


def _raw_sql(snapshot_dir, workload, candidate_ids, iterations):
    """Representative SQL+fetch probes; separate from the full public-method SLA."""
    placeholders = ",".join("?" for _ in candidate_ids)
    statements = {
        "category_price_probe": (
            "SELECT v.variant_id,ob.price_minor FROM variants v LEFT JOIN offers o USING(variant_id) "
            "LEFT JOIN current_offer_observations ob USING(offer_id) WHERE v.category=? "
            "ORDER BY v.variant_id LIMIT 20", [workload["category"]]),
        "filtered_fts_top5_before_group_dedup": (
            "SELECT e.evidence_id,e.variant_id,bm25(evidence_fts) AS score "
            "FROM evidence_fts JOIN evidence e ON e.id=evidence_fts.rowid "
            "WHERE evidence_fts MATCH ? AND e.variant_id IN (" + placeholders + ") "
            "ORDER BY score,e.evidence_id LIMIT 5", [workload["match"], *candidate_ids]),
        "resolve_provenance_probe": (
            "SELECT e.evidence_id,e.text,r.archive_path FROM evidence e JOIN raw_records r USING(raw_id) "
            "WHERE e.evidence_id IN (" + ",".join("?" for _ in workload["evidence_ids"]) + ")",
            workload["evidence_ids"]),
    }
    conn = _open(snapshot_dir / "catalog.sqlite")
    try:
        results = {}
        for name, (statement, parameters) in statements.items():
            plan = [dict(row) for row in conn.execute("EXPLAIN QUERY PLAN " + statement, parameters)]
            conn.execute(statement, parameters).fetchall()  # warm SQLite pages, no service result cache
            timings = []
            returned = 0
            for _ in range(iterations):
                started = time.perf_counter_ns()
                returned = len(conn.execute(statement, parameters).fetchall())
                timings.append((time.perf_counter_ns() - started) / 1_000_000)
            results[name] = {"calls": iterations, "p50_ms": statistics.median(timings),
                             "p95_ms": _percentile(timings, 0.95), "returned_rows": returned,
                             "sql": statement, "explain_query_plan": plan}
        return {"boundary": "SQL execute + fetchall; reused read-only connection, warm SQLite pages; no DTO/JSON",
                "not_the_total_sql_time_of_a_public_method": True, "probes": results}
    finally:
        conn.close()


def _measure_corpus(snapshots_root, snapshot_id, iterations):
    snapshot_dir = _snapshot_path(snapshots_root, snapshot_id)
    workload = _source_workload(snapshot_dir)
    phases, setups = {}, {}
    for name, size in (("warm_result_cache_disabled", 0), ("warm_result_cache_enabled", 512)):
        service = KnowledgeService(snapshots_root, cache_size=size, max_result_bytes=8192)
        operations, setup = _operations(service, snapshot_id, workload)
        setups[name] = setup
        serial, concurrent = {}, {}
        for path, operation in operations.items():
            for _ in range(3):
                _sample(operation)
            serial[path] = _summary([_sample(operation) for _ in range(iterations)])
            concurrent[path] = _summary(_concurrent(operation, iterations))
        phases[name] = {"serial": serial, "two_thread_closed_loop": concurrent}
    cold = {}
    for path in phases["warm_result_cache_disabled"]["serial"]:
        service = KnowledgeService(snapshots_root, cache_size=0, max_result_bytes=8192)
        setup = None
        if path == "retrieve_evidence":
            operations, setup = _operations(service, snapshot_id, workload)
            operation = operations[path]
        elif path == "search_products":
            operation = lambda: service.search_products(snapshot_id, {}, limit=5)
        elif path == "resolve_evidence":
            operation = lambda: service.resolve_evidence(snapshot_id, workload["evidence_ids"])
        else:
            operation = lambda: service.get_offer_facts(snapshot_id, workload["offer_ids"])
        cold[path] = {**_sample(operation), "candidate_setup": setup,
                      "definition": "fresh KnowledgeService first path call; OS file cache is not flushed"}
    inspector = KnowledgeService(snapshots_root, cache_size=0)
    initial = inspector.search_products(snapshot_id, {}, limit=1)
    ids = inspector.describe_candidate_set(snapshot_id, initial["candidate_set_id"])["variant_ids"]
    sql = _raw_sql(snapshot_dir, workload, ids, iterations)
    summaries = [value for phase in phases.values() for values in phase.values() for value in values.values()]
    serial = [value for phase in phases.values() for value in phase["serial"].values()]
    concurrent = [value for phase in phases.values() for value in phase["two_thread_closed_loop"].values()]
    return {
        "snapshot_id": snapshot_id, "counts": workload["counts"], "query": workload["query"],
        "request_scope": {"search_products": "first bounded page over complete catalog scope",
                          "retrieve_evidence": "top five over one previously built complete candidate set",
                          "resolve_evidence": "first bounded page for three evidence IDs; may return a cursor",
                          "get_offer_facts": "up to three offers; bounded response"},
        "iterations_per_path_per_warm_profile": iterations, "warmup_calls_per_path": 3,
        "candidate_setup": setups, "profiles": phases, "cold_service_calls": cold,
        "sql_without_service_result_cache": sql,
        "gates": {"warm_serial_p50_at_most_50ms": all(row["p50_ms"] <= 50 for row in serial),
                  "warm_serial_p95_at_most_50ms": all(row["p95_ms"] <= 50 for row in serial),
                  "two_thread_p95_at_most_100ms": all(row["p95_ms"] <= 100 for row in concurrent),
                  "all_measured_dtos_at_most_8192_bytes": all(row["max_response_bytes"] <= 8192 for row in summaries)
                      and all(row["response_bytes"] <= 8192 for row in cold.values()),
                  "byte_count_fields_match_serialization": all(row["declared_byte_count_mismatches"] == 0 for row in summaries)
                      and all(row["declared_bytes_match"] for row in cold.values())},
    }


def _synthetic_snapshot(source_dir, work_dir, scale):
    """Repeat evidence only; keep product catalog, observations and archives fixed."""
    work_dir.mkdir(parents=True)
    db_path = work_dir / "catalog.sqlite"
    source = _open(source_dir / "catalog.sqlite")
    target = sqlite3.connect(db_path)
    target.row_factory = sqlite3.Row
    try:
        source.backup(target)
        archives = list(source.execute("SELECT archive_path FROM raw_records"))
        for archive in archives:
            original = (source_dir / archive[0]).resolve()
            destination = (work_dir / archive[0]).resolve()
            if not original.is_relative_to(source_dir) or not destination.is_relative_to(work_dir):
                raise ValueError("Unsafe raw archive path in source snapshot")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        originals = [dict(row) for row in target.execute("SELECT * FROM evidence ORDER BY id")]
        columns = [name for name in originals[0] if name != "id"] if originals else []
        if not columns:
            raise ValueError("Cannot scale an empty evidence catalog")
        with target:
            if scale > 1:
                for record in originals:
                    # Existing review groups remain unchanged. Repeated singleton
                    # evidence receives one synthetic group, avoiding false novelty.
                    group = record["duplicate_group_id"] or "bench_group_" + sha256(record["evidence_id"].encode()).hexdigest()[:24]
                    target.execute("UPDATE evidence SET duplicate_group_id=? WHERE id=?", (group, record["id"]))
                    for repeat in range(1, scale):
                        copied = {**record, "duplicate_group_id": group,
                                  "evidence_id": "bench_" + sha256(f'{record["evidence_id"]}:{repeat}'.encode()).hexdigest()[:32]}
                        target.execute("INSERT INTO evidence(" + ",".join(columns) + ") VALUES(" +
                                       ",".join("?" for _ in columns) + ")", [copied[name] for name in columns])
            synthetic_id = "synthetic_" + sha256((source_dir.name + ":" + str(scale)).encode()).hexdigest()[:24]
            for key, value in (("dataset_id", synthetic_id), ("benchmark_source_snapshot_id", source_dir.name),
                               ("benchmark_synthetic_scale", str(scale)), ("benchmark_repeat_evidence_only", "true")):
                target.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, value))
        expected = len(originals) * scale
        if target.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] != expected:
            raise RuntimeError("Synthetic evidence scaling failed")
        if target.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] != expected:
            raise RuntimeError("Synthetic FTS triggers did not preserve evidence row counts")
    finally:
        source.close()
        target.close()
    frozen = export_snapshot(db_path, work_dir / "snapshots")
    verification = verify_snapshot(Path(frozen["path"]))
    if not verification["valid"]:
        raise RuntimeError("Synthetic snapshot failed complete verification")
    return frozen


def benchmark(snapshots_root: Path, snapshot_id: str, iterations=100, scale=10) -> dict:
    """Measure real snapshot and a disposable scaled-evidence stress snapshot."""
    if type(iterations) is not int or not 2 <= iterations <= 10000:
        raise ValueError("iterations must be an integer between 2 and 10000")
    if type(scale) is not int or not 1 <= scale <= 100:
        raise ValueError("scale must be an integer between 1 and 100")
    snapshots_root = Path(snapshots_root).resolve()
    source_dir = _snapshot_path(snapshots_root, snapshot_id)
    verification = verify_snapshot(source_dir)
    if not verification["valid"]:
        raise ValueError("Source snapshot failed complete verification")
    before = _tree_fingerprint(source_dir)
    actual = _measure_corpus(snapshots_root, snapshot_id, iterations)
    # Place disposable files in this repository's output directory, never beside
    # the immutable source snapshot or in the original data collection directory.
    scratch_root = Path(__file__).resolve().parents[1] / "output"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kb-benchmark-", dir=scratch_root) as directory:
        frozen = _synthetic_snapshot(source_dir, Path(directory) / "scaled", scale)
        synthetic_root = Path(frozen["path"]).parent
        synthetic = _measure_corpus(synthetic_root, frozen["snapshot_id"], iterations)
        synthetic["synthetic_snapshot_verified"] = True
        synthetic["temporary_snapshot_removed_after_measurement"] = True
    unchanged = _tree_fingerprint(source_dir) == before
    if not unchanged:
        raise RuntimeError("Immutable source snapshot changed during benchmark")
    return {
        "benchmark_version": "kb-local-dto-json-v1", "source_snapshot_id": snapshot_id,
        "runtime": {"python": sys.version, "implementation": platform.python_implementation(),
                    "sqlite": sqlite3.sqlite_version, "platform": platform.platform(),
                    "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "measurement_boundary": "KnowledgeService DTO construction + json.dumps(ensure_ascii=False,separators=(',',':')) + UTF-8 encoding",
        "excluded": ["API acquisition", "LLM/GPU", "agent/SDK file bridge", "Pydantic tool wrappers", "network transport"],
        "cache_notes": "Disabled means no service result cache; SQLite/OS pages and snapshot metadata may be warm. Cold means a fresh service, not a flushed OS cache.",
        "concurrency_notes": "Two closed-loop threads; call duration includes service locks and serialization, excluding executor queue delay.",
        "quantile_method": "p50 median; p95 nearest rank ceil(.95*n)",
        "synthetic_method": {"evidence_scale": scale, "source_products_unchanged": True,
                             "source_prices_unchanged": True, "new_independent_sources": 0,
                             "duplicate_policy": "existing groups retained; repeated singleton evidence shares a synthetic group",
                             "limitations": "Repeated evidence stress only, not ten times as many distinct products or diverse documents; listing/evidence joins may fan out."},
        "actual": actual, "synthetic": synthetic, "source_snapshot_unchanged": unchanged,
        "gates": {"actual": all(actual["gates"].values()), "synthetic": all(synthetic["gates"].values()),
                  "source_snapshot_unchanged": unchanged},
    }
