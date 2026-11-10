"""Incremental, restartable SerpApi acquisition. All budgets count HTTP attempts.

Planning is offline and read-only. Collection is called explicitly with a key; this
module never reads credential files. One SQLite ledger enforces a session's original
cap across restarts. An OS lock prevents concurrent collectors for the same data root.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

from .serpapi import (AuthenticationFailed, BudgetExceeded, QuotaExhausted,
                      SerpApiClient, now_iso)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_runs(data_root):
    """Read verified archives; never import or execute collection/deployment scripts."""
    root = Path(data_root).resolve() / "data" / "serpapi"
    documents, manifest = [], []
    for run in sorted(root.glob("run-*")):
        index = run / "index.jsonl"
        if not index.is_file():
            continue
        content = index.read_bytes()
        manifest.append({"run": run.name, "index_sha256": hashlib.sha256(content).hexdigest()})
        for line in content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or not record.get("raw_path"):
                continue
            raw = (run / record["raw_path"]).resolve()
            if not raw.is_relative_to(run.resolve()) or not raw.is_file():
                raise ValueError("Acquisition raw_path is missing or escapes its run")
            body = raw.read_bytes()
            if hashlib.sha256(body).hexdigest() != record.get("sha256"):
                raise ValueError(f"Raw hash mismatch for call {record.get('call_id')}")
            document = json.loads(body)
            if not isinstance(document, dict):
                raise ValueError("Archived acquisition JSON must be an object")
            documents.append((record, document))
    return documents, manifest


def _catalog(documents, config):
    queries = config["queries"]
    by_query = {entry["q"]: (i, entry["category"]) for i, entry in enumerate(queries)}
    token_ids, candidates = defaultdict(set), {}
    for record, document in documents:
        if record.get("engine") != "google_shopping":
            continue
        query = record.get("params", {}).get("q")
        for position, item in enumerate(document.get("shopping_results") or [], 1):
            pid = item.get("product_id")
            if pid is None:
                continue
            pid = str(pid)
            token = item.get("immersive_product_page_token")
            if token:
                token_ids[token].add(pid)
            if query not in by_query:
                continue
            query_order, category = by_query[query]
            rank = item.get("position", position)
            rank = rank if isinstance(rank, int) else position
            target = {"target_id": "target_" + _hash([category, query, pid])[:24],
                      "product_id": pid, "category": category, "query": query,
                      "query_order": query_order, "rank": rank,
                      "title": item.get("title", ""), "page_token": token,
                      "parent_call_id": record["call_id"],
                      "observed_at": record.get("requested_at", "")}
            key = (query, pid)
            # Prefer the latest archived occurrence; tie-breaking is deterministic.
            previous = candidates.get(key)
            if previous is None or (target["observed_at"], target["parent_call_id"]) > (previous["observed_at"], previous["parent_call_id"]):
                candidates[key] = target
    completed = set()
    for record, document in documents:
        if record.get("engine") != "google_product" or not document.get("product_results"):
            continue
        params = record.get("params", {})
        pid = record.get("product_id") or params.get("product_id")
        if pid is None:
            ids = token_ids.get(params.get("page_token"), set())
            pid = next(iter(ids)) if len(ids) == 1 else None
        if pid is not None:
            completed.add(str(pid))
    return list(candidates.values()), completed


def plan_collection(data_root: Path, config_path: Path, target_per_category=10) -> dict:
    """Plan N NEW detail successes per configured category, without any API request."""
    if isinstance(target_per_category, bool) or not isinstance(target_per_category, int) or target_per_category < 1:
        raise ValueError("target_per_category must be a positive integer")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    categories = list(dict.fromkeys(entry["category"] for entry in config["queries"]))
    if not categories:
        raise ValueError("No configured categories")
    documents, inputs = _read_runs(data_root)
    candidates, completed = _catalog(documents, config)
    by_query = defaultdict(list)
    used = set(completed)
    for candidate in sorted(candidates, key=lambda x: (x["query_order"], x["rank"], x["product_id"])):
        if candidate["product_id"] in used:
            continue
        by_query[candidate["query"]].append(candidate)
    queues = {query: deque(items) for query, items in by_query.items()}
    category_queries = {category: list(dict.fromkeys(entry["q"] for entry in config["queries"]
                        if entry["category"] == category)) for category in categories}
    query_turn = dict.fromkeys(categories, 0)
    def next_candidate(category):
        queries = category_queries[category]
        for step in range(len(queries)):
            position = (query_turn[category] + step) % len(queries)
            queue = queues.get(queries[position], deque())
            while queue and queue[0]["product_id"] in used:
                queue.popleft()
            if queue:
                candidate = queue.popleft()
                used.add(candidate["product_id"])
                query_turn[category] = (position + 1) % len(queries)
                return candidate
        return None
    targets = []
    for _ in range(target_per_category):
        for category in categories:
            candidate = next_candidate(category)
            if candidate is not None:
                targets.append(candidate)
    counts = dict(Counter(target["category"] for target in targets))
    identity = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "inputs": inputs, "target_per_category": target_per_category,
                "targets": [t["target_id"] for t in targets]}
    return {"plan_id": "plan_" + _hash(identity)[:24],
            "data_root": str(Path(data_root).resolve()), "config_path": str(config_path),
            "config_sha256": identity["config_sha256"],
            "target_per_category": target_per_category, "categories": categories,
            "planned_new_details": len(targets), "already_completed_details": len(completed),
            "category_counts": counts,
            "shortfalls": {c: target_per_category - counts.get(c, 0) for c in categories
                           if counts.get(c, 0) < target_per_category},
            "locale": config.get("locale", {"gl": "us", "hl": "en"}),
            "targets": targets, "input_manifest": inputs,
            "attempt_estimate_without_retries_or_refresh": len(targets),
            "budget_note": "Every HTTP attempt, including retries and query refreshes, consumes the session cap; provider billing may differ."}


@contextmanager
def _task_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another collector holds the data-root acquisition lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AttemptLedger:
    """Durable reservation before each call; crashed/incomplete attempts remain charged."""
    def __init__(self, path: Path, signature: dict, max_attempts: int, plan: dict | None = None):
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(
                call_id TEXT PRIMARY KEY, engine TEXT NOT NULL, requested_at TEXT NOT NULL,
                target_id TEXT, record_json TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS targets(
                target_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, target_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', last_error TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS refreshes(
                query TEXT PRIMARY KEY, status TEXT NOT NULL, mapping_json TEXT NOT NULL DEFAULT '{}',
                call_id TEXT);
        """)
        row = self.connection.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if row is None:
            if plan is None:
                raise ValueError("New ledger needs an acquisition plan")
            with self.connection:
                self.connection.executemany("INSERT INTO metadata VALUES(?,?)", [
                    ("signature", _json(signature)), ("max_attempts", str(max_attempts)),
                    ("plan", _json(plan)), ("created_at", now_iso())])
                self.connection.executemany("INSERT INTO targets(target_id,ordinal,target_json) VALUES(?,?,?)",
                    [(t["target_id"], i, _json(t)) for i, t in enumerate(plan["targets"])])
        elif json.loads(row[0]) != signature or int(self._value("max_attempts")) != max_attempts:
            self.connection.close()
            raise ValueError("Resume must use the original data root, config, target count and attempt cap")
        self.max_attempts = max_attempts

    def _value(self, key):
        return self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]

    @property
    def plan(self):
        return json.loads(self._value("plan"))

    @property
    def used(self):
        return self.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    def reserve(self, record):
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.used >= self.max_attempts:
                raise BudgetExceeded(f"Persistent session attempt cap {self.max_attempts} reached")
            self.connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)",
                (record.call_id, record.engine, record.requested_at, record.target_id,
                 _json(asdict(record)), "reserved"))
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def logged(self, record):
        with self.connection:
            self.connection.execute("UPDATE attempts SET status=?,record_json=? WHERE call_id=?",
                                    (record.status, _json(asdict(record)), record.call_id))

    def targets(self):
        return [(json.loads(row[0]), row[1]) for row in self.connection.execute(
            "SELECT target_json,status FROM targets ORDER BY ordinal")]

    def status(self, target, status, error=""):
        with self.connection:
            self.connection.execute("UPDATE targets SET status=?,last_error=? WHERE target_id=?",
                                    (status, error[:300], target["target_id"]))

    def reconcile(self, completed):
        # Handles a crash after the durable raw/index write and before marking complete.
        for target, _ in self.targets():
            if target["product_id"] in completed:
                self.status(target, "completed")

    def refresh_state(self, query):
        row = self.connection.execute("SELECT status,mapping_json,call_id FROM refreshes WHERE query=?", (query,)).fetchone()
        return (row[0], json.loads(row[1]), row[2]) if row else None

    def start_refresh(self, query):
        with self.connection:
            self.connection.execute("INSERT INTO refreshes(query,status) VALUES(?,'started')", (query,))

    def end_refresh(self, query, mapping, call_id, status):
        with self.connection:
            self.connection.execute("UPDATE refreshes SET status=?,mapping_json=?,call_id=? WHERE query=?",
                                    (status, _json(mapping), call_id, query))

    def close(self):
        self.connection.close()


def _needs_refresh(record, document):
    text = str((document or {}).get("error") or record.error or "").lower()
    return ("no results" in text or "hasn't returned any results" in text or
            ("token" in text and ("invalid" in text or "expired" in text)))


def _refresh_target(client, ledger, target):
    query = target["query"]
    state = ledger.refresh_state(query)
    if state is None:
        ledger.start_refresh(query)  # persist before the API call; never refresh twice after a crash
        record, data = client.search_shopping(query, record_meta={
            "category": target["category"], "parent_call_id": target["parent_call_id"],
            "target_id": target["target_id"]})
        mapping = {}
        if record.status == "ok":
            for row in (data or {}).get("shopping_results") or []:
                if row.get("product_id") is not None:
                    mapping[str(row["product_id"])] = {"page_token": row.get("immersive_product_page_token")}
        ledger.end_refresh(query, mapping, record.call_id, record.status)
        state = ledger.refresh_state(query)
    _, mapping, parent_call_id = state
    match = mapping.get(target["product_id"])
    if match is None:
        return None
    return {**target, "page_token": match.get("page_token"), "parent_call_id": parent_call_id}


def collect_incremental(data_root: Path, config_path: Path, session_dir: Path, key: str,
                        max_attempts=50, target_per_category=10) -> dict:
    """Collect only the persisted target plan; invoking again resumes the SAME cap.

    The caller supplies the credential without logging it. No account lookup or other
    implicit API call is made. Root orchestration owns any actual production invocation.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("A nonempty key is required")
    data_root, config_path = Path(data_root).resolve(), Path(config_path).resolve()
    session_dir = Path(session_dir).resolve()
    runs_root = data_root / "data" / "serpapi"
    if session_dir.parent != runs_root or not session_dir.name.startswith("run-"):
        raise ValueError("session_dir must be data_root/data/serpapi/run-<unique-name>")
    signature = {"data_root": str(data_root), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 "target_per_category": target_per_category}
    with _task_lock(runs_root / ".acquisition.lock"):
        session_dir.mkdir(parents=True, exist_ok=True)
        database = session_dir / "acquisition.sqlite"
        if not database.exists() and (session_dir / "index.jsonl").exists():
            raise ValueError("Existing run has no ledger; use a new session directory")
        initial_plan = None if database.exists() else plan_collection(data_root, config_path, target_per_category)
        ledger = AttemptLedger(database, signature, max_attempts, initial_plan)
        try:
            plan = ledger.plan
            _write_json(session_dir / "plan.json", plan)
            docs, _ = _read_runs(data_root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            _, completed = _catalog(docs, config)
            ledger.reconcile(completed)
            client = SerpApiClient(key, session_dir, max_attempts, locale=plan["locale"],
                                  attempt_guard=ledger.reserve, record_hook=ledger.logged)
            stopped = "finished"
            try:
                for target, status in ledger.targets():
                    if status == "completed":
                        continue
                    if ledger.used >= max_attempts:
                        raise BudgetExceeded(f"Persistent session attempt cap {max_attempts} reached")
                    def get_detail(item):
                        return client.get_product(item["product_id"], page_token=item.get("page_token"),
                            record_meta={"category": item["category"], "product_id": item["product_id"],
                                         "parent_call_id": item["parent_call_id"], "target_id": item["target_id"]})
                    record, document = get_detail(target)
                    if record.status != "ok" and _needs_refresh(record, document):
                        refreshed = _refresh_target(client, ledger, target)
                        if refreshed is not None:
                            record, document = get_detail(refreshed)
                    if record.status == "ok" and (document or {}).get("product_results"):
                        ledger.status(target, "completed")
                        completed.add(target["product_id"])
                    else:
                        ledger.status(target, "failed", record.error or "detail response contained no product_results")
            except BudgetExceeded:
                stopped = "budget_exhausted"
            except AuthenticationFailed:
                stopped = "authentication_failed"
            except QuotaExhausted:
                stopped = "quota_exhausted"
            states = ledger.targets()
            completed_targets = [t for t, status in states if status == "completed"]
            statuses = dict(Counter(status for _, status in states))
            if stopped == "finished" and statuses.get("failed"):
                stopped = "finished_with_failures"
            if stopped == "finished" and plan["shortfalls"]:
                stopped = "planning_shortfall"
            summary = {"session_dir": str(session_dir), "plan_id": plan["plan_id"],
                       "stopped": stopped, "max_attempts": max_attempts,
                       "attempts_used": ledger.used, "attempts_remaining": max_attempts - ledger.used,
                       "calls_made": ledger.used,
                       "attempts_this_invocation": client.calls_made,
                       "planned_new_details": plan["planned_new_details"],
                       "completed_new_details": len(completed_targets), "target_status_counts": statuses,
                       "completed_by_category": dict(Counter(t["category"] for t in completed_targets)),
                       "planning_shortfalls": plan["shortfalls"],
                       "billing_note": "Attempt reservations are conservative; no claim about actual provider credits."}
            _write_json(session_dir / "summary.json", summary)
            audit = {**summary, "created_at": now_iso(),
                     "targets": [{**t, "status": status} for t, status in states],
                     "attempt_status_counts": dict(ledger.connection.execute(
                         "SELECT status,COUNT(*) FROM attempts GROUP BY status")),
                     "refreshes": [dict(query=r[0], status=r[1], call_id=r[2]) for r in ledger.connection.execute(
                         "SELECT query,status,call_id FROM refreshes ORDER BY query")]}
            _write_json(session_dir / "acquisition_audit.json", audit)
            return summary
        finally:
            ledger.close()
