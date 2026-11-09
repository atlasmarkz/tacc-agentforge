"""Offline, incremental SQLite evidence store. No API calls or credentials are needed.

Money always uses integer minor units. ``attributes.numeric_value`` is a REAL
search index for non-monetary quantities only; ``normalized_number`` preserves
its Decimal representation. All source observations and evidence are immutable.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA_VERSION = "shopping-kb-2"
RETRIEVAL_VERSION = "fts5-unicode61-v1"
HERE = Path(__file__).resolve().parent
MINOR_DIGITS = {"USD": 2, "HKD": 2, "EUR": 2, "GBP": 2, "CAD": 2,
                "AUD": 2, "SGD": 2, "JPY": 0, "KRW": 0, "CNY": 2}
_TABLES = ("raw_records", "imports", "import_raw_records", "variants", "listings",
           "offers", "offer_observations", "evidence", "attributes", "reviews",
           "review_duplicate_groups", "category_schemas", "current_offer_rows",
           "current_attribute_rows", "current_listing_rows")
_SECRET_KEYS = {"api_key", "apikey", "access_token", "authorization", "password", "secret",
                "cookie", "set-cookie", "user_name", "username", "client_secret"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def stable(prefix, *parts):
    return prefix + "_" + digest(encode(parts).encode())[:24]


def plain(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = encode(value)
    return html.unescape(re.sub(r"<[^>]*>", " ", value)).strip()


def _normal(value):
    return re.sub(r"\s+", " ", plain(value).casefold()).strip()


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k.casefold() not in _SECRET_KEYS}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?i)((?:api_key|apikey|access_token|client_secret)=)[^&\s\"<>]+", r"\1[REDACTED]", value)
    return value


def clean_url(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        p = urlsplit(value)
        if p.scheme not in {"https", "http"} or not p.netloc or p.username or p.password:
            return None
        query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if k.casefold() not in _SECRET_KEYS]
        return urlunsplit((p.scheme, p.netloc, p.path, urlencode(query), p.fragment))
    except ValueError:
        return None


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Missing source requested_at; import time is not observation time")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid source requested_at") from exc
    if result.tzinfo is None:
        raise ValueError("Source requested_at must include a timezone")
    return result.astimezone(timezone.utc).isoformat(timespec="microseconds")


def currency_for(text, explicit=None):
    """A bare dollar sign or search locale is insufficient currency evidence."""
    text, explicit = str(text or "").strip(), str(explicit or "").upper()
    found = set()
    for symbol, code in [("US$", "USD"), ("HK$", "HKD"), ("CA$", "CAD"),
                         ("A$", "AUD"), ("S$", "SGD"), ("€", "EUR"), ("£", "GBP"), ("JP¥", "JPY")]:
        if re.search(r"(?<![A-Za-z])" + re.escape(symbol), text):
            found.add(code)
    for code in MINOR_DIGITS:
        if re.search(r"\b" + code + r"\b", text, re.I):
            found.add(code)
    if len(found) > 1:
        return None, "conflicting_currency"
    if found:
        return next(iter(found)), "explicit_price_symbol_or_code"
    if explicit in MINOR_DIGITS:
        if ("$" in text and explicit not in {"USD", "HKD", "CAD", "AUD", "SGD"}) or (
                "¥" in text and explicit not in {"JPY", "CNY"}):
            return None, "conflicting_currency"
        return explicit, "explicit_source_currency"
    return None, "unknown"


def money(value, currency):
    if value is None or isinstance(value, bool) or currency not in MINOR_DIGITS:
        return None
    try:
        number = Decimal(str(value))
        scaled = number * (10 ** MINOR_DIGITS[currency])
        if not number.is_finite() or number < 0 or scaled != scaled.to_integral_value():
            return None
        return int(scaled)
    except (InvalidOperation, ValueError):
        return None


def _price(row, code):
    value = row.get("extracted_price")
    if value is None:
        text = str(row.get("price") or "")
        match = re.fullmatch(r"\s*(?:(?:[A-Za-z]{1,3}\$?|[$€£¥])\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:[A-Za-z]{3})?\s*", text)
        value = match.group(1).replace(",", "") if match else None
    return money(value, code)


def shipping_cost(store, currency):
    text = plain(store.get("shipping"))
    conditions = _normal(store.get("details_and_offers"))
    # Thresholds, memberships, destination restrictions and qualifiers are not applied.
    qualifiers = r"\b(over|orders?|minimum|members?|membership|prime|eligible|selected|when|with|if|subscription|subscribe)\b"
    if re.search(qualifiers, _normal(text)) or ("free" in _normal(text) and re.search(qualifiers, conditions)):
        return None, "unknown"
    if _normal(text) in {"free", "free shipping", "free delivery"}:
        return 0, "known"
    ship_currency, _ = currency_for(text, currency)
    if currency is None or ship_currency != currency:
        return None, "unknown"
    value = store.get("shipping_extracted")
    if value is None:
        match = re.fullmatch(r"\+?\s*(?:(?:[A-Za-z]{1,3}\$?|[$€£¥])\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:[A-Za-z]{3})?\s*", text)
        value = match.group(1).replace(",", "") if match else None
    amount = money(value, currency)
    return (amount, "known") if amount is not None else (None, "unknown")


def _insert(conn, table, record):
    columns = list(record)
    conn.execute(f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                 [record[c] for c in columns])


def _metadata(conn, key, value):
    conn.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, str(value)))


def _connect(path):
    conn = sqlite3.connect(path, uri=isinstance(path, str) and path.startswith("file:"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _safe_child(root, relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError("Archive paths must be safe relative paths")
    path = (root / p).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes its root")
    return path


def _load_schemas(path):
    schemas = json.loads(path.read_text())
    categories = schemas.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Category schema requires a categories object")
    for category, schema in categories.items():
        if not isinstance(category, str) or not isinstance(schema.get("attributes", {}), dict):
            raise ValueError("Invalid category schema")
        for name, spec in schema.get("attributes", {}).items():
            if not name or spec.get("type") not in {"number", "boolean", "string"}:
                raise ValueError("Attribute schema must declare a supported type")
    return schemas


def _attribute(name, value, schema):
    result = {"canonical_name": _normal(name), "value_type": "unknown", "status": "unknown",
              "normalized_number": None, "numeric_value": None, "unit": None,
              "bool_value": None, "text_value": None}
    matched = None
    for key, spec in schema.get("attributes", {}).items():
        if _normal(name) in {_normal(key), *(_normal(x) for x in spec.get("aliases", []))}:
            result["canonical_name"], matched = key, spec
            break
    if matched is None:
        return result
    result["value_type"] = matched["type"]
    text = _normal(value)
    if text in {"", "unknown", "not specified", "n/a", "none specified"}:
        return result
    if matched["type"] == "string":
        result.update(status="known", text_value=text)
    elif matched["type"] == "boolean":
        truthy = {"yes", "true", "1", *(_normal(x) for x in matched.get("true_values", []))}
        falsy = {"no", "false", "0", *(_normal(x) for x in matched.get("false_values", []))}
        if text in truthy or text in falsy:
            result.update(status="known", bool_value=int(text in truthy))
    else:
        text = text.replace(",", "").replace("″", " in").replace('"', " in")
        # Strict single-value parsing: 'up to', intervals, mixed capacities stay unknown.
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([^\d]*)", text)
        if match:
            unit = match.group(2).strip() or matched.get("default_unit")
            units = {_normal(k): v for k, v in matched.get("units", {matched.get("unit", ""): "1"}).items()}
            factor = units.get(_normal(unit)) if unit is not None else None
            if factor is not None:
                try:
                    value = Decimal(match.group(1)) * Decimal(str(factor))
                    numeric = float(value)
                    if value.is_finite() and value >= 0 and abs(numeric) < float("inf"):
                        result.update(status="known", normalized_number=str(value), numeric_value=numeric, unit=matched.get("unit"))
                except (InvalidOperation, ValueError, OverflowError):
                    pass
    return result


def _explicit_pid(record, data):
    # Acquisition metadata is authoritative, including when a token maps to several IDs.
    for context in (record, record.get("metadata") or {}, record.get("acquisition_metadata") or {},
                    record.get("params") or {}, data.get("search_parameters") or {}):
        if context.get("product_id") is not None:
            return str(context["product_id"])
    return None


def _variant(conn, vid, pid, title, category, observed_at, source_url=None, code=None, basis="unknown", detail=False):
    _insert(conn, "variants", dict(variant_id=vid, provider_product_id=pid, title=plain(title), category=category,
        type="unknown", identity_status="provider_group_unverified" if pid else "unresolved",
        detail_status="details_available" if detail else "listing_only", source_url=source_url,
        currency=code, currency_basis=basis, observed_at=observed_at))
    # Replaying an older run must never replace the current summary.
    conn.execute("""UPDATE variants SET title=CASE WHEN ?<>'' THEN ? ELSE title END,
      category=CASE WHEN ?<>'unknown' THEN ? ELSE category END,
      source_url=COALESCE(?,source_url),currency=?,currency_basis=?,observed_at=?,
      detail_status=CASE WHEN ? THEN 'details_available' ELSE detail_status END
      WHERE variant_id=? AND observed_at<=?""",
      (plain(title), plain(title), category, category, source_url, code, basis, observed_at, detail, vid, observed_at))
    if detail:
        conn.execute("UPDATE variants SET detail_status='details_available' WHERE variant_id=?", (vid,))


def _group_reviews(conn):
    """Stable existing group IDs; new bridges merge via aliases and reject split leakage."""
    rows = conn.execute("SELECT review_id,text,evidence_id,duplicate_group_id,split FROM reviews ORDER BY review_id").fetchall()
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    shingles = []
    for row in rows:
        words = re.findall(r"\w+", _normal(row["text"]))
        shingles.append({tuple(words[j:j+5]) for j in range(len(words)-4)} if len(words) >= 12 else set())
    for i in range(len(rows)):
        for j in range(i):
            a, b = shingles[i], shingles[j]
            if _normal(rows[i]["text"]) == _normal(rows[j]["text"]) or (a and b and len(a & b) / len(a | b) >= .85):
                parents[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    existing = {r["group_id"]: r["split"] for r in conn.execute("SELECT * FROM review_duplicate_groups WHERE merged_into IS NULL")}
    for members in groups.values():
        ids = {rows[i]["duplicate_group_id"] for i in members}
        previous = sorted(ids & existing.keys())
        group = previous[0] if previous else min(ids)
        splits = {rows[i]["split"] for i in members if rows[i]["split"]}
        splits.update(existing[g] for g in previous if existing[g])
        if len(splits) > 1:
            raise ValueError("Near-duplicate review group would cross data splits")
        split = next(iter(splits), None)
        _insert(conn, "review_duplicate_groups", {"group_id": group, "split": split, "merged_into": None})
        conn.execute("UPDATE review_duplicate_groups SET split=? WHERE group_id=?", (split, group))
        for old in previous:
            if old != group:
                conn.execute("UPDATE review_duplicate_groups SET merged_into=? WHERE group_id=?", (group, old))
        for i in members:
            conn.execute("UPDATE reviews SET duplicate_group_id=?,split=? WHERE review_id=?", (group, split, rows[i]["review_id"]))
            conn.execute("UPDATE evidence SET duplicate_group_id=? WHERE evidence_id=?", (group, rows[i]["evidence_id"]))


def ingest_runs(run_dirs: list[Path], db_path: Path, config_path: Path,
                category_schema_path: Path | None = None) -> dict:
    """Verify archives, append source history, and update deterministic current projections.

    Reimporting the same archived response never creates another fact or changes its
    source timestamp. Existing raw archives are available for cross-run token joins.
    """
    db_path, config_path = Path(db_path), Path(config_path)
    category_schema_path = Path(category_schema_path) if category_schema_path else HERE.parent / "configs/category_schemas.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    schemas = _load_schemas(category_schema_path)
    category_by_query = {q["q"]: q["category"] for q in config.get("queries", [])}
    all_incoming, imports, failed_calls = [], [], 0
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        index_bytes = (run_dir / "index.jsonl").read_bytes()
        records = [json.loads(line) for line in index_bytes.splitlines() if line.strip()]
        incoming = []
        for record in records:
            if record.get("status") != "ok":
                failed_calls += 1
                continue
            path = _safe_child(run_dir, record.get("raw_path", ""))
            data_bytes = path.read_bytes()
            if digest(data_bytes) != record.get("sha256"):
                raise ValueError(f"SHA256 mismatch for call {record.get('call_id')}")
            observed = _timestamp(record.get("requested_at"))
            data = sanitize(json.loads(data_bytes))
            raw_id = stable("raw", record["sha256"])
            incoming.append((raw_id, sanitize(record), data, observed))
        provenance = {"run_name": run_dir.name, "records": sanitize(records),
                      "config": sanitize(config), "category_schemas_sha256": digest(encode(schemas).encode())}
        iid = stable("import", digest(index_bytes), digest(config_bytes), provenance["category_schemas_sha256"])
        imports.append((iid, digest(index_bytes), provenance, incoming))
        all_incoming.extend(incoming)
    if not all_incoming:
        raise ValueError("No successful archived responses")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    warnings = []
    try:
        conn.executescript((HERE / "schema.sql").read_text())
        previous_version = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if previous_version and previous_version[0] != SCHEMA_VERSION:
            raise ValueError("Database schema version differs; explicit migration required")
        with conn:
            for category, schema in schemas["categories"].items():
                conn.execute("INSERT INTO category_schemas VALUES(?,?) ON CONFLICT(category_id) DO UPDATE SET schema_json=excluded.schema_json",
                             (category, encode(schema)))
            existing_ids = {r[0] for r in conn.execute("SELECT raw_id FROM raw_records")}
            new = {}
            for raw_id, record, data, observed in sorted(all_incoming, key=lambda x: (x[3], x[0])):
                if raw_id in existing_ids or raw_id in new:
                    continue
                archive_path = f"raw/{raw_id}.json"
                archive_bytes = (encode(data) + "\n").encode()
                archive = _safe_child(db_path.parent, archive_path)
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists() and archive.read_bytes() != archive_bytes:
                    raise ValueError("Content-addressed archive collision")
                if not archive.exists():
                    archive.write_bytes(archive_bytes)
                params = sanitize(record.get("params") or {})
                # Preserve the acquisition identity metadata required for later imports.
                metadata = {k: record[k] for k in ("product_id", "category", "parent_call_id", "target_id", "metadata", "acquisition_metadata") if record.get(k) is not None}
                params["_acquisition"] = metadata
                _insert(conn, "raw_records", dict(raw_id=raw_id, call_id=str(record.get("call_id", "")),
                    engine=record.get("engine", "unknown"), origin_sha256=record["sha256"],
                    archive_sha256=digest(archive_bytes), archive_path=archive_path, observed_at=observed,
                    processed_at=record.get("processed_at"), params_json=encode(params)))
                new[raw_id] = (record, data, observed)
            for iid, index_hash, provenance, incoming in imports:
                _insert(conn, "imports", dict(import_id=iid, index_sha256=index_hash,
                    config_sha256=digest(config_bytes), provenance_json=encode(provenance),
                    imported_at=datetime.now(timezone.utc).isoformat(timespec="microseconds")))
                for raw_id, _, _, _ in incoming:
                    _insert(conn, "import_raw_records", {"import_id": iid, "raw_id": raw_id})
            # Join tokens across *all* archived runs. Explicit metadata takes priority.
            all_records = []
            token_map = defaultdict(set)
            explicit_codes = defaultdict(set)
            for raw in conn.execute("SELECT * FROM raw_records ORDER BY observed_at,raw_id"):
                params = json.loads(raw["params_json"])
                record = {"engine": raw["engine"], "params": params, **params.pop("_acquisition", {})}
                data = json.loads(_safe_child(db_path.parent, raw["archive_path"]).read_text())
                all_records.append((raw["raw_id"], record, data))
                if raw["engine"] == "google_shopping":
                    for row in data.get("shopping_results", []):
                        if row.get("product_id") is not None and row.get("immersive_product_page_token"):
                            token_map[row["immersive_product_page_token"]].add(str(row["product_id"]))
            def product_id(record, data):
                explicit = _explicit_pid(record, data)
                if explicit is not None:
                    return explicit
                pids = token_map.get((record.get("params") or {}).get("page_token"), set())
                return next(iter(pids)) if len(pids) == 1 else None
            for _, record, data in all_records:
                if record["engine"] == "google_product":
                    pid = product_id(record, data)
                    code = (data.get("product_results") or {}).get("price_insights", {}).get("currency")
                    if pid and code in MINOR_DIGITS:
                        explicit_codes[pid].add(code)
            def evidence(raw_id, vid, kind, title, text, pointer, url, label, scope, group=None):
                observed = new[raw_id][2]
                eid = stable("ev", raw_id, pointer, kind)
                _insert(conn, "evidence", dict(evidence_id=eid, variant_id=vid, kind=kind,
                    title=plain(title), text=plain(text), source_url=clean_url(url), source_label=plain(label),
                    raw_id=raw_id, json_pointer=pointer, observed_at=observed, scope=scope, duplicate_group_id=group))
                return eid
            # All new listings precede details so category hints and provider IDs are available.
            for raw_id, (record, data, observed) in sorted(new.items(), key=lambda x: (x[1][2], x[0])):
                if record.get("engine") != "google_shopping":
                    continue
                query = (record.get("params") or {}).get("q", "")
                category = record.get("category") or category_by_query.get(query, "unknown")
                for i, row in enumerate(data.get("shopping_results", [])):
                    pid = str(row["product_id"]) if row.get("product_id") is not None else None
                    vid = stable("variant", "google", pid) if pid else stable("variant", raw_id, i)
                    codes = explicit_codes.get(pid, set())
                    code, basis = currency_for(row.get("price"), next(iter(codes)) if len(codes) == 1 else row.get("currency"))
                    url = clean_url(row.get("product_link") or row.get("link"))
                    _variant(conn, vid, pid, row.get("title", ""), category, observed, url, code, basis)
                    pointer = f"/shopping_results/{i}"
                    _insert(conn, "listings", dict(listing_id=stable("listing", raw_id, i), variant_id=vid,
                        raw_id=raw_id, json_pointer=pointer, query=query, category_hint=category, source_url=url,
                        merchant=plain(row.get("source")), title=plain(row.get("title")), price_minor=_price(row, code),
                        currency=code, delivery_text=plain(row.get("delivery")), observed_at=observed))
                    evidence(raw_id, vid, "listing", row.get("title"),
                        f"Listing price: {row.get('price', 'unknown')}; seller: {row.get('source', 'unknown')}; delivery: {row.get('delivery', 'unknown')}. {plain(row.get('snippet'))}",
                        pointer, url, row.get("source"), "listing_unverified")
            for raw_id, (record, data, observed) in sorted(new.items(), key=lambda x: (x[1][2], x[0])):
                if record.get("engine") != "google_product":
                    continue
                p, params = data.get("product_results") or {}, record.get("params") or {}
                pid = product_id(record, data)
                vid = stable("variant", "google", pid) if pid else stable("variant", "unmatched_detail", raw_id)
                found = conn.execute("SELECT source_url,category FROM variants WHERE variant_id=?", (vid,)).fetchone()
                category = record.get("category") or (found["category"] if found else "unknown")
                source_url = found["source_url"] if found else None
                if pid is None:
                    warnings.append({"raw_id": raw_id, "issue": "detail_not_uniquely_linked_to_listing"})
                explicit = p.get("price_insights", {}).get("currency") or p.get("currency")
                code, basis = currency_for("", explicit)
                title = plain(p.get("title"))
                _variant(conn, vid, pid, title, category, observed, source_url, code, basis, detail=True)
                about = p.get("about_the_product") or {}
                if about.get("description"):
                    evidence(raw_id, vid, "description", title, about["description"], "/product_results/about_the_product/description",
                             about.get("link") or source_url, about.get("displayed_link") or "Google Product", "product_family_or_unknown")
                schema = schemas["categories"].get(category, {})
                attribute_sets = [(p.get("product_attributes") or [], "/product_results/product_attributes", "name"),
                                  (about.get("features") or [], "/product_results/about_the_product/features", "title")]
                for attrs, prefix, key in attribute_sets:
                    for i, attr in enumerate(attrs):
                        name, value = plain(attr.get(key)), plain(attr.get("value"))
                        if not name or not value:
                            continue
                        eid = evidence(raw_id, vid, "attribute", title, f"{name}: {value}", f"{prefix}/{i}",
                            about.get("link") or source_url, "Google Product", "product_family_or_unknown")
                        _insert(conn, "attributes", dict(attribute_id=stable("attr", eid), variant_id=vid,
                            name=_normal(name), value=value, evidence_id=eid, observed_at=observed, **_attribute(name, value, schema)))
                for i, store in enumerate(p.get("stores") or []):
                    scode, sbasis = currency_for(store.get("price"), store.get("currency") or explicit)
                    url = clean_url(store.get("link"))
                    options = store.get("selected_options") or store.get("options") or store.get("variant") or {}
                    store_title = plain(store.get("title"))
                    sku = store.get("sku") or store.get("product_id") or store.get("offer_id")
                    # Merchant URL alone is never a SKU: 143L/70L, colors and bundles remain separate.
                    oid = stable("offer", vid, str(store.get("merchant_id") or _normal(store.get("name"))),
                                 url, str(sku) if sku is not None else None, _normal(store_title), options,
                                 _normal(store.get("condition")))
                    _insert(conn, "offers", dict(offer_id=oid, variant_id=vid,
                        merchant_id=str(store["merchant_id"]) if store.get("merchant_id") is not None else None,
                        merchant=plain(store.get("name")) or "unknown", url=url,
                        identity_status="provider_group_unverified", store_title=store_title,
                        selected_options_json=encode(options)))
                    ship, ship_status = shipping_cost(store, scode)
                    eid = evidence(raw_id, vid, "offer", store_title or title,
                        f"Seller: {store.get('name')}; item price: {store.get('price')}; shipping: {store.get('shipping', 'unknown')}; additional costs: {store.get('additional_costs', 'unknown')}; displayed total (not a verified final total): {store.get('total')}; offer conditions: {plain(store.get('details_and_offers'))}; selected options: {plain(options)}",
                        f"/product_results/stores/{i}", url, store.get("name"), "offer_variant_unverified")
                    context = {"search_market": params.get("gl"), "language": params.get("hl"),
                               "destination": params.get("location"), "quantity": None,
                               "conditions": store.get("details_and_offers") or [], "shipping_scope": "unknown"}
                    cost = {"currency": scode, "price_minor": _price(store, scode), "shipping_minor": ship,
                            "shipping_status": ship_status, "tax_minor": None, "tax_status": "unknown",
                            "price_includes_tax": "unknown"}
                    _insert(conn, "offer_observations", dict(observation_id=stable("obs", eid), offer_id=oid,
                        raw_id=raw_id, evidence_id=eid, observed_at=observed, currency_basis=sbasis,
                        shipping_scope="unknown", shipping_text=plain(store.get("shipping")),
                        additional_costs=plain(store.get("additional_costs")), displayed_total_minor=money(store.get("extracted_total"), scode),
                        context_json=encode(context), context_key=stable("context", context), cost_signature=digest(encode(cost).encode()),
                        condition=plain(store.get("condition")) or None, store_title=store_title, **cost))
                for i, review in enumerate(p.get("user_reviews") or []):
                    text = plain(review.get("text"))
                    if not text:
                        continue
                    group = stable("dup", _normal(text))
                    eid = evidence(raw_id, vid, "review", review.get("title") or title, text,
                        f"/product_results/user_reviews/{i}", review.get("link") or source_url,
                        review.get("source"), "product_family_or_unknown", group)
                    rating = review.get("rating")
                    rating = rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else None
                    _insert(conn, "reviews", dict(review_id=stable("review", eid), variant_id=vid,
                        source_label=plain(review.get("source")), source_url=clean_url(review.get("link")),
                        title=plain(review.get("title")), text=text, rating=rating, date_text=plain(review.get("date")),
                        duplicate_group_id=group, split=None, scope="product_family_or_unknown", evidence_id=eid))
            _group_reviews(conn)
            # Normalized fields are versioned derivations, not original evidence.
            # Reparse when schemas evolve so existing archives need no API refetch.
            previous_schema = conn.execute("SELECT value FROM metadata WHERE key='category_schemas_sha256'").fetchone()
            schema_hash = digest(encode(schemas).encode())
            if previous_schema is None or previous_schema[0] != schema_hash:
                attribute_rows = conn.execute("SELECT a.attribute_id,a.name,a.value,v.category FROM attributes a JOIN variants v USING(variant_id)").fetchall()
                for attr in attribute_rows:
                    typed = _attribute(attr["name"], attr["value"], schemas["categories"].get(attr["category"], {}))
                    conn.execute("UPDATE attributes SET " + ",".join(k+"=?" for k in typed) + " WHERE attribute_id=?",
                                 [*typed.values(), attr["attribute_id"]])
            _metadata(conn, "category_schemas_sha256", schema_hash)
            for target, source in (("current_offer_rows", "_computed_current_offer_observations"),
                                   ("current_attribute_rows", "_computed_current_attributes"),
                                   ("current_listing_rows", "_computed_current_listings")):
                conn.execute(f"DELETE FROM {target}")
                conn.execute(f"INSERT INTO {target} SELECT * FROM {source}")
            raw_hashes = [r[0] for r in conn.execute("SELECT origin_sha256 FROM raw_records ORDER BY origin_sha256")]
            _metadata(conn, "schema_version", SCHEMA_VERSION)
            _metadata(conn, "schema_sha256", digest((HERE / "schema.sql").read_bytes()))
            _metadata(conn, "importer_sha256", digest(Path(__file__).read_bytes()))
            _metadata(conn, "python_version", sys.version.split()[0])
            _metadata(conn, "sqlite_version", sqlite3.sqlite_version)
            _metadata(conn, "retrieval_version", RETRIEVAL_VERSION)
            _metadata(conn, "tokenizer", "unicode61")
            _metadata(conn, "dataset_id", stable("dataset", raw_hashes, schemas, SCHEMA_VERSION))
            _metadata(conn, "as_of", conn.execute("SELECT MAX(observed_at) FROM raw_records").fetchone()[0])
            _metadata(conn, "mode", "persistent_archive")
            _metadata(conn, "identity_policy", "Provider groups and merchant variants remain provisional; source attributes may be family-scoped")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Foreign-key integrity check failed")
    finally:
        conn.close()
    report = audit_database(db_path)
    report.update(imported_raw_records=len(new), failed_calls=failed_calls, warnings=warnings)
    return report


def audit_database(db_path: Path) -> dict:
    conn = _connect(f"file:{Path(db_path).resolve()}?mode=ro")
    try:
        counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _TABLES}
        age = conn.execute("SELECT MIN(observed_at),MAX(observed_at) FROM raw_records").fetchone()
        conflicts = [dict(r) for r in conn.execute("""SELECT variant_id,canonical_name,COUNT(DISTINCT value) AS distinct_values
              FROM current_attributes GROUP BY variant_id,canonical_name HAVING COUNT(DISTINCT value)>1""")]
        return {"counts": counts, "category_counts": dict(conn.execute("SELECT category,COUNT(*) FROM variants GROUP BY category")),
                "dataset_id": dict(conn.execute("SELECT key,value FROM metadata")).get("dataset_id"),
                "observed_range": {"oldest": age[0], "newest": age[1]},
                "unknown_listing_currency": conn.execute("SELECT COUNT(*) FROM listings WHERE currency IS NULL").fetchone()[0],
                "unknown_shipping_offers": conn.execute("SELECT COUNT(*) FROM current_offer_observations WHERE shipping_status='unknown'").fetchone()[0],
                "unknown_tax_offers": conn.execute("SELECT COUNT(*) FROM current_offer_observations WHERE tax_status='unknown'").fetchone()[0],
                "current_cost_conflicts": conn.execute("SELECT COUNT(*) FROM (SELECT offer_id,context_key FROM current_offer_observations WHERE cost_conflict=1 GROUP BY offer_id,context_key)").fetchone()[0],
                "attribute_conflicts": conflicts,
                "duplicate_review_rows": counts["reviews"] - conn.execute("SELECT COUNT(DISTINCT duplicate_group_id) FROM reviews").fetchone()[0],
                "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
                "foreign_key_errors": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
                "fts_rowid_errors": conn.execute("SELECT COUNT(*) FROM evidence e LEFT JOIN evidence_fts f ON f.rowid=e.id WHERE f.evidence_id IS NULL OR f.evidence_id<>e.evidence_id").fetchone()[0],
                "limitations": ["Archived prices are not a live-price guarantee.", "Tax is unknown; displayed totals never establish affordability.",
                    "Shipping destination, quantity and order grouping remain unresolved.", "Provider groups and store variants require identity review.",
                    "Family-scoped attributes and reviews do not establish exact-SKU suitability."]}
    finally:
        conn.close()


def _fingerprint(conn):
    h = hashlib.sha256()
    for table in _TABLES:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        h.update(table.encode())
        h.update(encode(sorted(rows, key=encode)).encode())
    metadata = {r[0]: r[1] for r in conn.execute("SELECT key,value FROM metadata") if r[0] not in {"snapshot_id", "mode"}}
    h.update(encode(metadata).encode())
    return h.hexdigest()


def export_snapshot(db_path: Path, snapshots_root: Path) -> dict:
    """Freeze a transaction-consistent DB plus all referenced sanitized raw archives."""
    db_path, snapshots_root = Path(db_path).resolve(), Path(snapshots_root).resolve()
    snapshots_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshots_root))
    source, target = None, None
    try:
        source = _connect(f"file:{db_path}?mode=ro")
        target = _connect(temp / "catalog.sqlite")
        source.backup(target)
        source.close()
        source = None
        fingerprint = _fingerprint(target)
        snapshot_id = "snapshot_" + fingerprint[:24]
        with target:
            _metadata(target, "snapshot_id", snapshot_id)
            _metadata(target, "mode", "offline_snapshot")
        provenance = [json.loads(r[0]) for r in target.execute("SELECT provenance_json FROM imports ORDER BY import_id")]
        schemas = {r[0]: json.loads(r[1]) for r in target.execute("SELECT category_id,schema_json FROM category_schemas ORDER BY category_id")}
        for row in target.execute("SELECT archive_path,archive_sha256 FROM raw_records"):
            original = _safe_child(db_path.parent, row[0])
            data = original.read_bytes()
            if digest(data) != row[1]:
                raise ValueError("Source raw archive digest mismatch")
            destination = _safe_child(temp, row[0])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        metadata = dict(target.execute("SELECT key,value FROM metadata"))
        target.close()
        target = None
        (temp / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        (temp / "category_schemas.json").write_text(json.dumps({"schema_version": "1", "categories": schemas}, ensure_ascii=False, indent=2) + "\n")
        (temp / "schema.sql").write_bytes((HERE / "schema.sql").read_bytes())
        report = audit_database(temp / "catalog.sqlite")
        (temp / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        files = {p.relative_to(temp).as_posix(): digest(p.read_bytes()) for p in sorted(temp.rglob("*")) if p.is_file()}
        manifest = {"snapshot_id": snapshot_id, "dataset_id": metadata["dataset_id"], "catalog_fingerprint": fingerprint,
                    "schema_version": SCHEMA_VERSION, "retrieval_version": RETRIEVAL_VERSION, "metadata": metadata,
                    "files": files, "near_duplicate_policy": "5-word shingles Jaccard >= .85; stable group aliases; split conflicts rejected"}
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        output = snapshots_root / snapshot_id
        if output.exists():
            verification = verify_snapshot(output)
            if not verification["valid"]:
                raise ValueError("Existing immutable snapshot failed verification")
            shutil.rmtree(temp)
        else:
            temp.rename(output)
        return {"snapshot_id": snapshot_id, "path": str(output), "snapshot_path": str(output),
                "database_path": str(output / "catalog.sqlite"), "dataset_id": metadata["dataset_id"], "counts": report["counts"]}
    finally:
        if source is not None:
            source.close()
        if target is not None:
            target.close()
        if temp.exists():
            shutil.rmtree(temp)


def _resolve_json_pointer(document, pointer):
    """Resolve a JSON Pointer (RFC 6901), rejecting invalid escapes and indexes.

    This accepts the JSON string representation, not URI fragments. A dictionary
    token such as ``01`` is a valid key; the same token is not an array index.
    """
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("JSON pointer must be empty or start with '/'")
    value = document
    for token in pointer.split("/")[1:] if pointer else []:
        if re.search(r"~(?![01])", token):
            raise ValueError("JSON pointer has an invalid '~' escape")
        # Decode ~1 first: '~01' denotes the literal key '~1', not '/'.
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                raise ValueError("JSON pointer dictionary key does not exist")
            value = value[token]
        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("JSON pointer array index is invalid")
            index = int(token)
            if index >= len(value):
                raise ValueError("JSON pointer array index is out of range")
            value = value[index]
        else:
            raise ValueError("JSON pointer traverses a scalar value")
    return value


def verify_snapshot(snapshot_dir: Path) -> dict:
    """Explicit, full verification; normal retrieval does not re-hash every file."""
    root = Path(snapshot_dir).resolve()
    errors = []
    pointers_checked = 0
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        expected = manifest["files"]
        for name, expected_hash in expected.items():
            path = _safe_child(root, name)
            if not path.is_file() or digest(path.read_bytes()) != expected_hash:
                errors.append(f"File missing or digest mismatch: {name}")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(expected) | {"manifest.json"}:
            errors.append("Snapshot contains unexpected or missing files")
        if "catalog.sqlite" not in expected or "provenance.json" not in expected:
            errors.append("Manifest omits required catalog or provenance")
        if errors:
            return {"valid": False, "snapshot_id": manifest.get("snapshot_id"), "errors": errors,
                    "evidence_pointers_checked": pointers_checked}
        conn = _connect(f"file:{root / 'catalog.sqlite'}?mode=ro")
        try:
            metadata = dict(conn.execute("SELECT key,value FROM metadata"))
            if metadata.get("snapshot_id") != manifest["snapshot_id"] or metadata.get("dataset_id") != manifest.get("dataset_id"):
                errors.append("Manifest/catalog identity mismatch")
            if _fingerprint(conn) != manifest.get("catalog_fingerprint"):
                errors.append("Logical catalog fingerprint mismatch")
            if manifest["snapshot_id"] != "snapshot_" + manifest["catalog_fingerprint"][:24]:
                errors.append("Snapshot ID does not match catalog fingerprint")
            raw_cache = {}
            for raw in conn.execute("SELECT archive_path,archive_sha256 FROM raw_records"):
                if expected.get(raw[0]) != raw[1]:
                    errors.append(f"Raw provenance digest mismatch: {raw[0]}")
                try:
                    # Every referenced archive is decoded exactly once even when
                    # hundreds of evidence rows point into the same response.
                    raw_cache[raw[0]] = json.loads(_safe_child(root, raw[0]).read_text())
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"Invalid raw JSON archive {raw[0]}: {exc}")
            for evidence in conn.execute("""SELECT e.evidence_id,e.json_pointer,r.archive_path
                    FROM evidence e LEFT JOIN raw_records r USING(raw_id) ORDER BY e.id"""):
                pointers_checked += 1
                archive_path = evidence["archive_path"]
                if archive_path not in raw_cache:
                    errors.append(f"Evidence raw archive unavailable: {evidence['evidence_id']}")
                    continue
                try:
                    _resolve_json_pointer(raw_cache[archive_path], evidence["json_pointer"])
                except (ValueError, TypeError) as exc:
                    errors.append(f"Evidence pointer invalid: {evidence['evidence_id']}: {exc}")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
                errors.append("Database integrity check failed")
            if conn.execute("SELECT COUNT(*) FROM evidence e LEFT JOIN evidence_fts f ON f.rowid=e.id WHERE f.evidence_id IS NULL OR f.evidence_id<>e.evidence_id").fetchone()[0]:
                errors.append("FTS evidence rowid mismatch")
        finally:
            conn.close()
        return {"valid": not errors, "snapshot_id": manifest["snapshot_id"], "errors": errors,
                "files_checked": len(expected), "evidence_pointers_checked": pointers_checked}
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        return {"valid": False, "errors": [str(exc)], "evidence_pointers_checked": pointers_checked}
