"""Read-only, bounded shopping knowledge queries over immutable SQLite snapshots.

No agent framework or HTTP dependency is required. Snapshot selection is a host
decision; an agent adapter obtains it from the trusted CallContext.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time


class KnowledgeError(ValueError):
    def __init__(self, code, public_message, *, recoverable=False):
        self.code, self.public_message, self.recoverable = code, public_message, recoverable
        super().__init__(public_message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError):
        return None


def _invalid(message):
    raise KnowledgeError("invalid_arguments", message)


def _explicit_specs(text):
    """Detect only explicit unit-bearing identity conflicts, never infer specs."""
    found = defaultdict(set)
    units = {
        "mah": ("charge", Decimal(1)), "wh": ("energy", Decimal(1)),
        "mb": ("storage", Decimal(1)), "gb": ("storage", Decimal(1000)), "tb": ("storage", Decimal(1000000)),
        "mib": ("binary_storage", Decimal(1)), "gib": ("binary_storage", Decimal(1024)), "tib": ("binary_storage", Decimal(1048576)),
        "l": ("volume", Decimal(1)), "liter": ("volume", Decimal(1)), "liters": ("volume", Decimal(1)),
        "litre": ("volume", Decimal(1)), "litres": ("volume", Decimal(1)), "ml": ("volume", Decimal("0.001")),
        "mm": ("length", Decimal(1)), "cm": ("length", Decimal(10)), "inch": ("length", Decimal("25.4")),
        "inches": ("length", Decimal("25.4")), "in": ("length", Decimal("25.4")), '"': ("length", Decimal("25.4")),
        "″": ("length", Decimal("25.4")),
    }
    pattern = r'(?<![\d.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mAh|Wh|GiB|TiB|MiB|GB|TB|MB|ml|liters?|litres?|L|cm|mm|inches|inch|in\b|″|")(?![A-Za-z])'
    for amount, unit in re.findall(pattern, str(text), flags=re.IGNORECASE):
        family, scale = units[unit.casefold()]
        found[family].add(Decimal(amount.replace(",", "")) * scale)
    return found


def _variant_conflict(expected, observed):
    expected_specs, observed_specs = _explicit_specs(expected), _explicit_specs(observed)
    return sorted(key for key in expected_specs.keys() & observed_specs.keys()
                  if expected_specs[key].isdisjoint(observed_specs[key]))


@dataclass(frozen=True)
class _Snapshot:
    snapshot_id: str
    dataset_id: str
    as_of: str
    db: Path
    fingerprint: tuple
    metadata: dict
    categories: dict


class KnowledgeService:
    """SQLite service with an LRU of immutable facts, never cached price ages."""

    def __init__(self, snapshots_root: Path, cache_size=512, max_result_bytes=8192, price_ttl_hours=24):
        self.snapshots_root = Path(snapshots_root).expanduser().resolve()
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            _invalid("cache_size must be a nonnegative integer")
        if not isinstance(max_result_bytes, int) or isinstance(max_result_bytes, bool) or not 1024 <= max_result_bytes <= 65536:
            _invalid("max_result_bytes must be between 1024 and 65536")
        try:
            ttl = float(price_ttl_hours)
        except (ValueError, TypeError):
            _invalid("price_ttl_hours must be positive")
        if not 0 < ttl < float("inf"):
            _invalid("price_ttl_hours must be positive")
        self.cache_size, self.max_result_bytes, self.price_ttl_hours = cache_size, max_result_bytes, ttl
        self._cache, self._sets, self._snapshots = OrderedDict(), OrderedDict(), {}
        self._lock, self._local = threading.RLock(), threading.local()

    @contextmanager
    def query_context(self, *, deadline=None, cancellation=None):
        """Install a monotonic deadline and threading.Event for a worker call."""
        previous = getattr(self._local, "control", None)
        self._local.control = (deadline, cancellation)
        try:
            self._check()
            yield
        finally:
            self._local.control = previous

    def _check(self):
        control = getattr(self._local, "control", None)
        if control:
            deadline, cancellation = control
            if cancellation is not None and cancellation.is_set():
                raise KnowledgeError("cancelled", "Knowledge query was cancelled.", recoverable=True)
            if deadline is not None and time.monotonic() >= deadline:
                raise KnowledgeError("deadline_exceeded", "Knowledge query exceeded its deadline.", recoverable=True)

    def _open(self, path):
        connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        def progress():
            try:
                self._check()
                return 0
            except KnowledgeError:
                return 1
        connection.set_progress_handler(progress, 500)
        return connection

    def _snapshot(self, snapshot_id):
        self._check()
        if not isinstance(snapshot_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", snapshot_id):
            _invalid("snapshot_id must be an identifier, not a filesystem path")
        folder = (self.snapshots_root / snapshot_id).resolve()
        if folder.parent != self.snapshots_root:
            _invalid("snapshot_id must select a snapshot inside the configured root")
        path = (folder / "catalog.sqlite").resolve()
        if path.parent != folder or not path.is_file():
            raise KnowledgeError("unknown_snapshot", "The selected snapshot is unavailable.")
        stat = path.stat()
        fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._snapshots.get(snapshot_id)
            if cached:
                if cached.fingerprint != fingerprint:
                    raise KnowledgeError("snapshot_changed", "An immutable snapshot changed; publish a new snapshot ID.")
                return cached
            connection = self._open(path)
            try:
                rows = connection.execute(
                    "SELECT 'metadata' AS kind,key AS name,value FROM metadata UNION ALL "
                    "SELECT 'category',category_id,schema_json FROM category_schemas"
                ).fetchall()
            finally:
                connection.close()
            metadata = {r["name"]: r["value"] for r in rows if r["kind"] == "metadata"}
            categories = {r["name"]: json.loads(r["value"]) for r in rows if r["kind"] == "category"}
            if metadata.get("snapshot_id") != snapshot_id:
                raise KnowledgeError("snapshot_mismatch", "Snapshot metadata does not match its selected ID.")
            if _time(metadata.get("as_of")) is None:
                raise KnowledgeError("invalid_snapshot", "Snapshot has no valid as_of timestamp.")
            cached = _Snapshot(snapshot_id, metadata.get("dataset_id", snapshot_id), metadata["as_of"],
                               path, fingerprint, metadata, categories)
            self._snapshots[snapshot_id] = cached
            return cached

    @contextmanager
    def _connection(self, snapshot):
        self._check()
        connection = self._open(snapshot.db)
        try:
            yield connection
            self._check()
        except sqlite3.OperationalError as exc:
            self._check()
            raise KnowledgeError("query_failed", "The knowledge query could not be completed.") from exc
        finally:
            connection.close()

    def _get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def _put(self, key, value):
        if not self.cache_size:
            return
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _remember_set(self, snapshot, frames, filter_digest):
        selected = sorted((x["variant_id"], x["selected_price"]["source_record_id"] if x["selected_price"] else None)
                          for x in frames)
        set_id = "cand_" + _hash([snapshot.snapshot_id, filter_digest, selected])[:32]
        value = {"snapshot_id": snapshot.snapshot_id, "variant_ids": tuple(x["variant_id"] for x in frames),
                 "selected_price_record_ids": tuple(x[1] for x in selected if x[1]),
                 "filter_digest": filter_digest}
        with self._lock:
            self._sets[set_id] = value
            self._sets.move_to_end(set_id)
            while len(self._sets) > max(32, self.cache_size):
                self._sets.popitem(last=False)
        return set_id

    def describe_candidate_set(self, snapshot_id, candidate_set_id):
        """Host-only scope inspection; not registered as a model tool."""
        self._snapshot(snapshot_id)
        with self._lock:
            value = self._sets.get(candidate_set_id)
            if not value or value["snapshot_id"] != snapshot_id:
                raise KnowledgeError("unknown_candidate_set", "Search products again to obtain an active candidate set.", recoverable=True)
            self._sets.move_to_end(candidate_set_id)
            return deepcopy(value)

    def describe(self, snapshot_id):
        snapshot = self._snapshot(snapshot_id)
        return {"snapshot_id": snapshot_id, "dataset_id": snapshot.dataset_id, "as_of": snapshot.as_of,
                "schema_version": snapshot.metadata.get("schema_version"),
                "retrieval_version": snapshot.metadata.get("retrieval_version"),
                "categories": deepcopy(snapshot.categories)}

    def _envelope(self, snapshot, **payload):
        now = datetime.now(timezone.utc)
        result = {"snapshot_id": snapshot.snapshot_id, "dataset_id": snapshot.dataset_id, "as_of": snapshot.as_of,
                  "responded_at": now.isoformat(timespec="seconds"), "bytes": 0,
                  "truncated": False, "next_cursor": None, **deepcopy(payload)}
        def age(value):
            if isinstance(value, dict):
                if "price_minor" in value and "observed_at" in value:
                    observed = _time(value["observed_at"])
                    seconds = max(0, int((now - observed).total_seconds())) if observed else None
                    value["price_age_seconds"] = seconds
                    value["price_stale"] = seconds > self.price_ttl_hours * 3600 if seconds is not None else None
                for child in value.values():
                    age(child)
            elif isinstance(value, list):
                for child in value:
                    age(child)
        age(result)
        self._measure(result)
        return result

    @staticmethod
    def _measure(result):
        for _ in range(5):
            count = len(_json(result).encode())
            if result["bytes"] == count:
                return count
            result["bytes"] = count
        return len(_json(result).encode())

    def _fit(self, result):
        if self._measure(result) > self.max_result_bytes:
            raise KnowledgeError("result_budget_exceeded", "Requested facts exceed the configured result budget; request fewer records.", recoverable=True)
        return result

    @staticmethod
    def _cursor(value):
        return base64.urlsafe_b64encode(_json(value).encode()).decode().rstrip("=")

    @staticmethod
    def _read_cursor(cursor, fingerprint):
        if cursor is None:
            return 0, 0
        try:
            if not isinstance(cursor, str) or len(cursor) > 2048:
                raise ValueError()
            data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
            index, offset = data["i"], data.get("o", 0)
            if data["f"] != fingerprint or type(index) is not int or type(offset) is not int or min(index, offset) < 0:
                raise ValueError()
            return index, offset
        except (ValueError, TypeError, KeyError, UnicodeDecodeError):
            _invalid("cursor does not belong to this query")

    @staticmethod
    def _limit(limit, maximum=20):
        if type(limit) is not int or not 1 <= limit <= maximum:
            _invalid(f"limit must be an integer between 1 and {maximum}")

    def _filters(self, snapshot, filters):
        if not isinstance(filters, dict):
            _invalid("filters must be an object")
        allowed = {"category", "type", "currency", "max_price_minor", "min_price_minor", "variant_ids", "hard_constraints", "include_unknown"}
        if set(filters) - allowed:
            _invalid("Unknown filter fields: " + ", ".join(sorted(set(filters) - allowed)))
        filters = deepcopy(filters)
        category = filters.get("category")
        if category is not None and not isinstance(category, str):
            _invalid("category must be a string")
        if category is not None and category not in snapshot.categories:
            _invalid("category is not defined by this snapshot's category schemas")
        currency = filters.get("currency")
        if currency is not None and (not isinstance(currency, str) or not re.fullmatch("[A-Z]{3}", currency)):
            _invalid("currency must be an uppercase three-letter code")
        for key in ("max_price_minor", "min_price_minor"):
            if key in filters and (type(filters[key]) is not int or not 0 <= filters[key] <= 9_000_000_000_000_000):
                _invalid(key + " must be a nonnegative integer")
            if key in filters and currency is None:
                _invalid("price filters require an explicit currency")
        if "include_unknown" in filters and type(filters["include_unknown"]) is not bool:
            _invalid("include_unknown must be a boolean")
        if "type" in filters and (not isinstance(filters["type"], str) or len(filters["type"]) > 100):
            _invalid("type must be a short string")
        if "variant_ids" in filters:
            self._ids(filters["variant_ids"], maximum=10000)
            filters["variant_ids"] = sorted(set(filters["variant_ids"]))
        constraints = filters.get("hard_constraints", {})
        if not isinstance(constraints, dict) or len(constraints) > 20:
            _invalid("hard_constraints must contain at most 20 attributes")
        schemas = snapshot.categories.get(category, {}).get("attributes", {}) if category else {}
        for key, requirement in constraints.items():
            if not isinstance(key, str) or not isinstance(requirement, dict) or set(requirement) - {"op", "value", "unit"}:
                _invalid("Each hard constraint needs op, value, and optional unit")
            if requirement.get("op") not in ("eq", "gte", "lte", "gt", "lt", "contains") or "value" not in requirement:
                _invalid("Unsupported hard-constraint operation")
            definitions = [schemas[key]] if key in schemas else [c.get("attributes", {}).get(key) for c in snapshot.categories.values()]
            definitions = [d for d in definitions if d]
            if not definitions:
                _invalid("Unknown schema attribute: " + key)
            types = {d.get("type") for d in definitions}
            value = requirement["value"]
            if types == {"number"}:
                if isinstance(value, bool):
                    _invalid("Numeric attributes need a finite numeric value")
                try:
                    number = Decimal(str(value))
                    if not number.is_finite():
                        raise InvalidOperation()
                except (InvalidOperation, ValueError, TypeError):
                    _invalid("Numeric attributes need a finite numeric value")
                requirement["value"] = str(number)
                if requirement["op"] == "contains":
                    _invalid("contains is not valid for a numeric attribute")
            elif types == {"boolean"} and (type(value) is not bool or requirement["op"] != "eq"):
                _invalid("Boolean attributes require eq with true or false")
            elif types == {"string"} and (not isinstance(value, str) or requirement["op"] not in ("eq", "contains")):
                _invalid("String attributes require eq or contains with a string")
            if "unit" in requirement and not isinstance(requirement["unit"], str):
                _invalid("unit must be a string")
        return filters

    @staticmethod
    def _ids(values, maximum=100):
        if not isinstance(values, list) or len(values) > maximum or any(
            not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", x) for x in values
        ):
            _invalid(f"IDs must be a list of at most {maximum} identifiers")

    @staticmethod
    def _fact(row):
        row = dict(row)
        try:
            context = json.loads(row.pop("context_json") or "{}")
        except (ValueError, TypeError):
            context = {}
        try:
            options = json.loads(row.pop("selected_options_json") or "{}")
        except (ValueError, TypeError):
            options = {}
        expected_title = row.pop("variant_title")
        conflicts = _variant_conflict(expected_title, row["store_title"] + " " + _json(options))
        row["variant_conflict"] = bool(conflicts)
        row["variant_conflict_fields"] = conflicts
        row["selected_options"] = options
        row["context"] = {k: context.get(k) for k in ("destination", "quantity", "search_market")}
        row["unknown_fields"] = [key for key in ("currency", "price_minor") if row.get(key) is None]
        if not context.get("destination"):
            row["unknown_fields"].append("destination")
        if type(context.get("quantity")) is not int or context["quantity"] < 1:
            row["unknown_fields"].append("quantity")
        for name in ("shipping", "tax"):
            if row.get(name + "_status") != "known" or row.get(name + "_minor") is None:
                row["unknown_fields"].append(name)
        if row.get("shipping_scope") not in ("item", "order"):
            row["unknown_fields"].append("shipping_scope")
        if row.get("price_includes_tax") not in ("included", "excluded"):
            row["unknown_fields"].append("price_includes_tax")
        if row.get("cost_conflict"):
            row["unknown_fields"].append("cost_conflict")
        if conflicts:
            row["unknown_fields"].append("variant_conflict")
        if row.get("identity_status") != "verified":
            row["unknown_fields"].append("variant_identity")
        row["complete_total_minor"] = None
        # A validator additionally checks destination, quantity and requirements.
        if not row["unknown_fields"] and row["kind"] == "offer":
            row["complete_total_minor"] = row["price_minor"] + row["shipping_minor"] + (
                row["tax_minor"] if row["price_includes_tax"] == "excluded" else 0)
        return row

    def _price_sql(self, where):
        return (
            "SELECT 'offer' AS kind,ob.observation_id AS source_record_id,o.variant_id,o.offer_id,"
            "ob.observation_id,ob.evidence_id,ob.observed_at,ob.currency,ob.currency_basis,ob.price_minor,"
            "ob.shipping_minor,ob.shipping_status,ob.shipping_scope,ob.tax_minor,ob.tax_status,"
            "ob.price_includes_tax,ob.context_key,ob.context_json,ob.cost_conflict,o.identity_status,"
            "o.store_title,o.selected_options_json,v.title AS variant_title FROM current_offer_observations ob "
            "JOIN offers o USING(offer_id) JOIN variants v ON v.variant_id=o.variant_id WHERE " + where
        )

    def _catalog(self, snapshot, filters):
        key = ("catalog", snapshot.snapshot_id, _json(filters), snapshot.metadata.get("schema_version"))
        cached = self._get(key)
        if cached is not None:
            return cached
        clauses, parameters = ["1=1"], []
        for field in ("category", "type"):
            if field in filters:
                clauses.append(field + "=?")
                parameters.append(filters[field])
        if "variant_ids" in filters:
            ids = filters["variant_ids"]
            clauses.append("variant_id IN (" + ",".join("?" for _ in ids) + ")" if ids else "0=1")
            parameters.extend(ids)
        with self._connection(snapshot) as connection:
            variants = [dict(x) for x in connection.execute(
                "SELECT variant_id,title,category,type,identity_status,detail_status FROM variants WHERE " +
                " AND ".join(clauses) + " ORDER BY variant_id", parameters)]
            ids = [x["variant_id"] for x in variants]
            if not ids:
                self._put(key, [])
                return []
            placeholders = ",".join("?" for _ in ids)
            prices = self._price_sql("o.variant_id IN (" + placeholders + ")") + (
                " UNION ALL SELECT 'listing',l.listing_id,l.variant_id,NULL,NULL,e.evidence_id,l.observed_at,"
                "l.currency,'listing',l.price_minor,NULL,'unknown','unknown',NULL,'unknown','unknown',"
                "'listing','{}',l.cost_conflict,'provider_group_unverified',l.title,'{}',v.title FROM current_listings l "
                "JOIN variants v ON v.variant_id=l.variant_id JOIN evidence e ON "
                "e.raw_id=l.raw_id AND e.json_pointer=l.json_pointer AND e.variant_id=l.variant_id AND e.kind='listing' "
                "AND e.evidence_id=(SELECT MIN(e2.evidence_id) FROM evidence e2 WHERE e2.raw_id=l.raw_id "
                "AND e2.json_pointer=l.json_pointer AND e2.variant_id=l.variant_id AND e2.kind='listing') "
                "WHERE l.variant_id IN (" + placeholders + ")"
            )
            facts = defaultdict(list)
            for row in connection.execute(prices, ids + ids):
                facts[row["variant_id"]].append(self._fact(row))
            attributes = defaultdict(lambda: defaultdict(list))
            for row in connection.execute(
                "SELECT variant_id,canonical_name,value_type,status,normalized_number,bool_value,text_value,unit,evidence_id "
                "FROM current_attributes WHERE variant_id IN (" + placeholders + ") ORDER BY variant_id,canonical_name,evidence_id", ids
            ):
                attributes[row["variant_id"]][row["canonical_name"]].append(dict(row))
        frames = []
        for variant in variants:
            self._check()
            available = facts[variant["variant_id"]]
            attr = attributes[variant["variant_id"]]
            expected_attributes = self._identity_attributes(attr)
            for fact in available:
                self._extra_conflicts(fact, expected_attributes)
            eligible = [p for p in available if p["currency"] == filters.get("currency", p["currency"])
                        and p["price_minor"] is not None and not p["cost_conflict"] and not p["variant_conflict"] and p["currency"] is not None]
            if "currency" not in filters and len({p["currency"] for p in eligible}) > 1:
                eligible = []  # No implicit exchange rate or mixed-currency minimum.
            detailed = [p for p in eligible if p["kind"] == "offer"]
            selected = min(detailed or eligible, key=lambda p: (p["price_minor"], p["source_record_id"]), default=None)
            statuses = {}
            price = selected["price_minor"] if selected else None
            for field, compare in (("max_price_minor", lambda a, b: a <= b), ("min_price_minor", lambda a, b: a >= b)):
                if field in filters:
                    statuses[field] = "unknown" if price is None else ("satisfied" if compare(price, filters[field]) else "violated")
            for name, requirement in filters.get("hard_constraints", {}).items():
                statuses[name] = self._constraint(attr.get(name, []), requirement, snapshot.categories.get(variant["category"], {}).get("attributes", {}).get(name, {}))
            if "violated" in statuses.values() or (not filters.get("include_unknown", True) and "unknown" in statuses.values()):
                continue
            shown_attrs = []
            names = list(filters.get("hard_constraints", {})) or sorted(attr)[:3]
            for name in names:
                values = attr.get(name, [])
                if values:
                    a = values[0]
                    shown_attrs.append({"name": name, "value": a["normalized_number"] if a["value_type"] == "number" else
                                        bool(a["bool_value"]) if a["value_type"] == "boolean" and a["bool_value"] is not None else a["text_value"],
                                        "unit": a["unit"], "evidence_id": a["evidence_id"],
                                        "status": "unknown" if len({_json([v["normalized_number"], v["bool_value"], v["text_value"], v["unit"]]) for v in values}) > 1 else a["status"]})
            frames.append({**variant, "title": variant["title"][:160], "selected_price": selected,
                           "hard_constraint_status": statuses, "attributes": shown_attrs,
                           "constraint_status": "unknown" if "unknown" in statuses.values() else "satisfied"})
        frames.sort(key=lambda x: (x["selected_price"] is None,
                                  x["selected_price"]["currency"] if x["selected_price"] else "",
                                  x["selected_price"]["price_minor"] if x["selected_price"] else 0, x["variant_id"]))
        self._put(key, frames)
        return frames

    @staticmethod
    def _identity_attributes(attributes):
        values = []
        for rows in attributes.values():
            known = [a for a in rows if a["status"] == "known" and a["value_type"] == "number"]
            signatures = {(a["normalized_number"], a["unit"]) for a in known}
            if len(signatures) == 1:
                amount, unit = next(iter(signatures))
                values.append(str(amount) + " " + str(unit))
        return " ".join(values)

    @staticmethod
    def _extra_conflicts(fact, expected_attributes):
        extra = _variant_conflict(expected_attributes, fact["store_title"] + " " + _json(fact["selected_options"]))
        if extra:
            fact["variant_conflict"] = True
            fact["variant_conflict_fields"] = sorted(set(fact["variant_conflict_fields"]) | set(extra))
            if "variant_conflict" not in fact["unknown_fields"]:
                fact["unknown_fields"].append("variant_conflict")
            fact["complete_total_minor"] = None

    @staticmethod
    def _constraint(rows, requirement, definition):
        known = [a for a in rows if a["status"] == "known"]
        signatures = {_json([a["normalized_number"], a["bool_value"], a["text_value"], a["unit"]]) for a in known}
        if not known or len(signatures) != 1:
            return "unknown"
        value = known[0]
        wanted, op = requirement["value"], requirement["op"]
        try:
            if value["value_type"] == "number":
                actual, wanted = Decimal(value["normalized_number"]), Decimal(str(wanted))
                requested_unit = requirement.get("unit", definition.get("unit"))
                if requested_unit != value["unit"]:
                    multiplier = definition.get("units", {}).get(requested_unit)
                    if multiplier is None or value["unit"] != definition.get("unit"):
                        return "unknown"
                    wanted *= Decimal(str(multiplier))
            elif value["value_type"] == "boolean":
                actual = bool(value["bool_value"])
            elif value["value_type"] == "string":
                actual, wanted = value["text_value"].casefold(), str(wanted).casefold()
            else:
                return "unknown"
            passed = {"eq": lambda: actual == wanted, "gte": lambda: actual >= wanted,
                      "lte": lambda: actual <= wanted, "gt": lambda: actual > wanted,
                      "lt": lambda: actual < wanted, "contains": lambda: wanted in actual}[op]()
            return "satisfied" if passed else "violated"
        except (InvalidOperation, TypeError, ValueError, AttributeError):
            return "unknown"

    def search_products(self, snapshot_id, filters: dict, limit=5, cursor=None):
        self._limit(limit)
        snapshot = self._snapshot(snapshot_id)
        filters = self._filters(snapshot, filters)
        frames = self._catalog(snapshot, filters)
        digest = _hash(filters)
        set_id = self._remember_set(snapshot, frames, digest)
        fingerprint = _hash([snapshot_id, set_id, "catalog"])
        start, offset = self._read_cursor(cursor, fingerprint)
        if offset or start > len(frames):
            _invalid("cursor is outside this candidate set")
        end = min(len(frames), start + limit)
        while True:
            next_cursor = self._cursor({"f": fingerprint, "i": end}) if end < len(frames) else None
            result = self._envelope(snapshot, candidate_set_id=set_id, total_count=len(frames), scope_complete=True,
                                    candidates=frames[start:end], filter_digest=digest,
                                    truncated=end < len(frames), next_cursor=next_cursor)
            if self._measure(result) <= self.max_result_bytes:
                return result
            if end <= start + 1:
                return self._fit(result)
            end -= 1

    def retrieve_evidence(self, snapshot_id, query, candidate_set_id, k=5, mode="filtered"):
        self._limit(k, 20)
        snapshot = self._snapshot(snapshot_id)
        candidate_set = self.describe_candidate_set(snapshot_id, candidate_set_id)
        if mode not in ("filtered", "keyword"):
            _invalid("mode must be filtered or keyword")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            _invalid("query must contain 1 to 2000 characters")
        aliases = {"耳机": "earbuds", "降噪": "noise cancelling", "充电器": "charger", "充电宝": "power bank", "键盘": "keyboard", "背包": "backpack", "防水": "waterproof"}
        expanded = query + " " + " ".join(v for key, v in aliases.items() if key in query)
        terms = list(dict.fromkeys(re.findall(r"\w+(?:[-']\w+)*", expanded)))
        if not terms or len(terms) > 100:
            _invalid("query must contain between 1 and 100 searchable terms")
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        ids = candidate_set["variant_ids"]
        key = ("evidence", snapshot_id, candidate_set_id, match, mode, k, snapshot.metadata.get("retrieval_version"))
        evidence = self._get(key)
        if evidence is None:
            evidence, seen = [], set()
            if mode == "keyword" or ids:
                where, parameters = "evidence_fts MATCH ?", [match]
                if mode == "filtered":
                    where += " AND e.variant_id IN (" + ",".join("?" for _ in ids) + ")"
                    parameters.extend(ids)
                with self._connection(snapshot) as connection:
                    # Filtering precedes top-k. Cursor iteration has no arbitrary recall cap.
                    rows = connection.execute(
                        "SELECT e.evidence_id,e.variant_id,e.kind,e.title,e.observed_at,e.scope,e.duplicate_group_id,"
                        "bm25(evidence_fts) AS score FROM evidence_fts JOIN evidence e ON e.id=evidence_fts.rowid "
                        "WHERE " + where + " ORDER BY score,e.evidence_id", parameters)
                    ranked = []
                    for row in rows:
                        group = row["duplicate_group_id"] or row["evidence_id"]
                        if group not in seen:
                            ranked.append(dict(row)); seen.add(group)
                        if len(ranked) >= k:
                            break
                    if ranked:
                        texts = {r["evidence_id"]: r["text"] for r in connection.execute(
                            "SELECT evidence_id,text FROM evidence WHERE evidence_id IN (" + ",".join("?" for _ in ranked) + ")",
                            [r["evidence_id"] for r in ranked])}
                        for row in ranked:
                            text = texts[row["evidence_id"]]
                            row["title"] = row["title"][:120]
                            row["text"] = text[:420]
                            row["text_truncated"] = len(text) > 420
                        evidence = ranked
            self._put(key, evidence)
        result = self._envelope(snapshot, candidate_set_id=candidate_set_id, mode=mode, query=query[:200],
                                evidence=evidence, requested_k=k, returned_count=len(evidence))
        # Preserve every selected top-k ID. Excerpts can be resolved separately;
        # silently dropping IDs here would change measured retrieval recall.
        while self._measure(result) > self.max_result_bytes:
            changed = False
            for item in result["evidence"]:
                if len(item["text"]) > 16:
                    item["text"] = item["text"][:max(16, len(item["text"]) // 2)]
                    item["text_truncated"] = True
                    changed = True
                elif len(item["title"]) > 24:
                    item["title"] = item["title"][:max(24, len(item["title"]) // 2)]
                    item["title_truncated"] = True
                    changed = True
            result["truncated"] = True
            if not changed:
                return self._fit(result)
        if any(item["text_truncated"] for item in result["evidence"]):
            result["truncated"] = True
        return self._fit(result)

    def get_offer_facts(self, snapshot_id, offer_ids):
        self._ids(offer_ids)
        snapshot = self._snapshot(snapshot_id)
        ids = sorted(set(offer_ids))
        key = ("offers", snapshot_id, tuple(ids))
        facts = self._get(key)
        if facts is None:
            with self._connection(snapshot) as connection:
                facts = [self._fact(row) for row in connection.execute(
                    self._price_sql("o.offer_id IN (" + ",".join("?" for _ in ids) + ")") + " ORDER BY o.offer_id,ob.context_key,ob.observation_id", ids)] if ids else []
                variant_ids = sorted({f["variant_id"] for f in facts})
                attributes = defaultdict(lambda: defaultdict(list))
                if variant_ids:
                    for row in connection.execute(
                        "SELECT variant_id,canonical_name,value_type,status,normalized_number,unit FROM current_attributes "
                        "WHERE variant_id IN (" + ",".join("?" for _ in variant_ids) + ")", variant_ids):
                        attributes[row["variant_id"]][row["canonical_name"]].append(dict(row))
                for fact in facts:
                    self._extra_conflicts(fact, self._identity_attributes(attributes[fact["variant_id"]]))
            self._put(key, facts)
        found = {f["offer_id"] for f in facts}
        groups = defaultdict(list)
        for fact in facts:
            groups[fact["offer_id"]].append(fact)
        ordered = sorted(groups)
        chosen = []
        result = self._envelope(snapshot, offers=[], unknown_ids=[x for x in ids if x not in found], remaining_offer_ids=[])
        for index, offer_id in enumerate(ordered):
            proposed = chosen + groups[offer_id]
            candidate = self._envelope(snapshot, offers=proposed, unknown_ids=result["unknown_ids"],
                                       remaining_offer_ids=ordered[index + 1:], truncated=index + 1 < len(ordered))
            if self._measure(candidate) > self.max_result_bytes:
                if not chosen:
                    raise KnowledgeError("result_budget_exceeded", "One complete offer context group exceeds the result budget; select a specific context through the host.", recoverable=True)
                result = self._envelope(snapshot, offers=chosen, unknown_ids=result["unknown_ids"],
                                        remaining_offer_ids=ordered[index:], truncated=True)
                break
            chosen, result = proposed, candidate
        return self._fit(result)

    def resolve_evidence(self, snapshot_id, evidence_ids, cursor=None):
        self._ids(evidence_ids)
        snapshot = self._snapshot(snapshot_id)
        ids = sorted(set(evidence_ids))
        fingerprint = _hash([snapshot_id, ids, "resolve"])
        start, offset = self._read_cursor(cursor, fingerprint)
        if start > len(ids):
            _invalid("cursor is outside this evidence request")
        key = ("resolve", snapshot_id, tuple(ids))
        records = self._get(key)
        if records is None:
            with self._connection(snapshot) as connection:
                records = {r["evidence_id"]: dict(r) for r in connection.execute(
                    "SELECT e.evidence_id,e.variant_id,e.kind,e.title,e.text,e.source_url,e.source_label,e.observed_at,e.scope,"
                    "e.raw_id,e.json_pointer,r.archive_path,r.archive_sha256,r.origin_sha256 FROM evidence e JOIN raw_records r USING(raw_id) "
                    "WHERE e.evidence_id IN (" + ",".join("?" for _ in ids) + ")", ids)} if ids else {}
            self._put(key, records)
        evidence, index, position = [], start, offset
        while index < len(ids):
            original = records.get(ids[index])
            if original is None:
                index += 1; position = 0
                continue
            if position > len(original["text"]):
                _invalid("cursor is outside this evidence text")
            record = deepcopy(original)
            stop = min(position + 1400, len(record["text"]))
            record["text"] = record["text"][position:stop]
            record.update(text_start=position, text_end=stop, text_total_chars=len(original["text"]), text_truncated=stop < len(original["text"]))
            next_index, next_position = (index, stop) if stop < len(original["text"]) else (index + 1, 0)
            next_cursor = self._cursor({"f": fingerprint, "i": next_index, "o": next_position}) if next_index < len(ids) else None
            result = self._envelope(snapshot, evidence=evidence + [record], unknown_ids=[x for x in ids if x not in records],
                                    next_cursor=next_cursor, truncated=next_cursor is not None)
            if self._measure(result) > self.max_result_bytes:
                if evidence:
                    break
                if stop > position:
                    # Reduce a single Unicode excerpt without cutting UTF-8 bytes.
                    while self._measure(result) > self.max_result_bytes and stop > position:
                        stop = position + (stop - position) // 2
                        record["text"] = original["text"][position:stop]
                        record.update(text_end=stop, text_truncated=stop < len(original["text"]))
                        result["evidence"] = [record]
                        result["next_cursor"] = self._cursor({"f": fingerprint, "i": index, "o": stop})
                        result["truncated"] = True
                    if stop == position:
                        raise KnowledgeError("result_budget_exceeded", "Evidence metadata leaves no room for one text character; increase the result budget.", recoverable=True)
                    return self._fit(result)
                return self._fit(result)
            evidence.append(record)
            index, position = next_index, next_position
            if next_position:
                return result
        next_cursor = self._cursor({"f": fingerprint, "i": index, "o": position}) if index < len(ids) else None
        return self._fit(self._envelope(snapshot, evidence=evidence, unknown_ids=[x for x in ids if x not in records],
                                        next_cursor=next_cursor, truncated=next_cursor is not None))
