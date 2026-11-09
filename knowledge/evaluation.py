"""Offline structural development labels and controlled retrieval comparisons.

These automatically generated labels are deliberately narrow: an explicit,
current attribute assertion for one provider product group. They are not human
relevance judgments, held-out task scores, faithfulness scores, or price checks.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from .service import KnowledgeService


LABEL_VERSION = "structural-development-attributes-v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _tree_fingerprint(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): _hash_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _open(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _snapshot_path(snapshots_root, snapshot_id):
    if not isinstance(snapshot_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", snapshot_id):
        raise ValueError("snapshot_id must be an identifier")
    root = Path(snapshots_root).resolve()
    folder = (root / snapshot_id).resolve()
    if folder.parent != root or not (folder / "catalog.sqlite").is_file():
        raise ValueError("Snapshot is unavailable inside snapshots_root")
    return folder


def retrieval_metrics(retrieved_ids, relevant_ids, duplicate_groups=None, k=5):
    """Group-aware Recall@K and MRR@K, deduplicating before the K cutoff.

    IDs without a duplicate group form singleton groups. Empty gold has undefined
    recall/MRR and is reported as None instead of silently scoring a success.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    groups = duplicate_groups or {}
    group = lambda item: groups.get(item) or item
    relevant = {group(item) for item in relevant_ids}
    ranked, seen = [], set()
    for item in retrieved_ids:
        value = group(item)
        if value not in seen:
            ranked.append(value)
            seen.add(value)
        if len(ranked) == k:
            break
    hits = len(set(ranked) & relevant)
    first = next((index for index, item in enumerate(ranked, 1) if item in relevant), None)
    return {
        "recall_at_5" if k == 5 else f"recall_at_{k}": hits / len(relevant) if relevant else None,
        "mrr_at_5" if k == 5 else f"mrr_at_{k}": (1 / first if first else 0.0) if relevant else None,
        "relevant_groups": len(relevant), "retrieved_groups": len(ranked), "hit_groups": hits,
    }


def _attribute_signature(row):
    return (row["value_type"], row["normalized_number"], row["bool_value"], row["text_value"], row["unit"])


def prepare_development_labels(snapshot_dir: Path, output_path: Path) -> dict:
    """Write ten deterministic SQL-derived cases before any retriever is queried.

    Selection is round-robin across categories, then provider groups and fields.
    Every gold set contains *all current explicit attribute records* matching the
    same variant, canonical field, typed value and unit. Conflicting fields and
    unknown values are excluded. No generator, retriever, judge or network runs.
    """
    snapshot_dir, output_path = Path(snapshot_dir).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("Refusing to replace frozen development labels")
    if output_path.is_relative_to(snapshot_dir):
        raise ValueError("Labels must be written outside the immutable snapshot")
    before = _tree_fingerprint(snapshot_dir)
    db_path = snapshot_dir / "catalog.sqlite"
    conn = _open(db_path)
    try:
        metadata = dict(conn.execute("SELECT key,value FROM metadata"))
        grouped = defaultdict(list)
        for row in conn.execute(
            "SELECT a.*,v.category,v.title AS product_title,e.duplicate_group_id "
            "FROM current_attributes a JOIN variants v USING(variant_id) "
            "JOIN evidence e ON e.evidence_id=a.evidence_id "
            "ORDER BY v.category,a.variant_id,a.canonical_name,a.evidence_id"
        ):
            grouped[(row["category"], row["variant_id"], row["canonical_name"])].append(dict(row))
        buckets = defaultdict(list)
        for (category, variant_id, canonical), rows in grouped.items():
            signatures = {_attribute_signature(row) for row in rows if row["status"] == "known"}
            if len(signatures) != 1 or any(row["status"] != "known" for row in rows):
                continue
            selected = rows[0]
            if selected["value_type"] not in {"number", "boolean", "string"}:
                continue
            value = (selected["normalized_number"] if selected["value_type"] == "number" else
                     bool(selected["bool_value"]) if selected["value_type"] == "boolean" else selected["text_value"])
            if value is None:
                continue
            requirement = {"op": "eq", "value": value}
            if selected["unit"] is not None:
                requirement["unit"] = selected["unit"]
            relevant = [row["evidence_id"] for row in rows]
            buckets[category].append({
                "category": category, "variant_id": variant_id,
                "query": f'{selected["name"]} {selected["value"]}',
                "filters": {"category": category, "variant_ids": [variant_id],
                            "hard_constraints": {canonical: requirement}, "include_unknown": False},
                "predicate": {"canonical_name": canonical, "value_type": selected["value_type"],
                              "normalized_number": selected["normalized_number"], "bool_value": selected["bool_value"],
                              "text_value": selected["text_value"], "unit": selected["unit"]},
                "relevant_evidence_ids": relevant,
                "relevant_group_ids": sorted({row["duplicate_group_id"] or row["evidence_id"] for row in rows}),
                "reference_scope": "current explicit attribute assertions for one provider group; exact SKU remains unverified",
            })
        # Spread each category over distinct provider groups before revisiting one.
        for category, items in buckets.items():
            by_variant = defaultdict(list)
            for item in items:
                by_variant[item["variant_id"]].append(item)
            spread = []
            while any(by_variant.values()):
                for variant in sorted(by_variant):
                    if by_variant[variant]:
                        spread.append(by_variant[variant].pop(0))
            buckets[category] = spread
        cases = []
        while len(cases) < 10 and any(buckets.values()):
            for category in sorted(buckets):
                if buckets[category] and len(cases) < 10:
                    case = buckets[category].pop(0)
                    case["case_id"] = f"development-{len(cases) + 1:02d}"
                    cases.append(case)
        if len(cases) != 10:
            raise ValueError("Snapshot needs at least ten unambiguous, schema-typed attribute cases")
        payload = {
            "label_version": LABEL_VERSION, "split": "development",
            "label_kind": "programmatic_structural_not_human_relevance",
            "snapshot_id": metadata["snapshot_id"], "dataset_id": metadata["dataset_id"],
            "catalog_sha256": _hash_file(db_path),
            "snapshot_manifest_sha256": _hash_file(snapshot_dir / "manifest.json"),
            "generation": {"method": "independent SQL; same variant + canonical field + equal typed value and unit",
                           "retriever_queried": False, "llm_judge_used": False,
                           "gold_scope": "all matching current attribute evidence; duplicate groups count once",
                           "category_counts": {category: sum(c["category"] == category for c in cases)
                                               for category in sorted({c["category"] for c in cases})}},
            "cases": cases,
            "limitations": ["Not human relevance labels or held-out evaluation.",
                            "Other semantically useful descriptions/reviews are outside this structural gold definition.",
                            "No answer generation, faithfulness, preference fit, feasibility or arithmetic is measured."],
        }
    finally:
        conn.close()
    if _tree_fingerprint(snapshot_dir) != before:
        raise RuntimeError("Snapshot changed during label preparation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"labels_path": str(output_path), "labels_sha256": _hash_file(output_path),
            "case_count": len(cases), "category_counts": payload["generation"]["category_counts"],
            "label_kind": payload["label_kind"], "snapshot_unchanged": True}


def _resolved_provenance(service, snapshot_id, evidence_ids, snapshot_dir):
    if not evidence_ids:
        return set()
    covered, cursor = set(), None
    while True:
        result = service.resolve_evidence(snapshot_id, evidence_ids, cursor=cursor)
        for evidence in result["evidence"]:
            relative = evidence.get("archive_path")
            path = (snapshot_dir / relative).resolve() if isinstance(relative, str) else None
            if (path is not None and path.is_relative_to(snapshot_dir) and path.is_file()
                    and evidence.get("raw_id") and str(evidence.get("json_pointer", "")).startswith("/")
                    and _hash_file(path) == evidence.get("archive_sha256")):
                covered.add(evidence["evidence_id"])
        cursor = result.get("next_cursor")
        if not cursor:
            return covered


def evaluate(snapshots_root: Path, snapshot_id: str, labels_path: Path) -> dict:
    """Compare keyword/filtered evidence with exactly the same candidate/price IDs."""
    snapshot_dir = _snapshot_path(snapshots_root, snapshot_id)
    before = _tree_fingerprint(snapshot_dir)
    label_bytes = Path(labels_path).read_bytes()
    labels = json.loads(label_bytes)
    if labels.get("label_version") != LABEL_VERSION or labels.get("snapshot_id") != snapshot_id:
        raise ValueError("Labels must select this snapshot and the supported structural development format")
    if labels.get("catalog_sha256") != _hash_file(snapshot_dir / "catalog.sqlite"):
        raise ValueError("Labels were frozen against another catalog")
    if labels.get("snapshot_manifest_sha256") != _hash_file(snapshot_dir / "manifest.json"):
        raise ValueError("Labels were frozen against another snapshot manifest")
    conn = _open(snapshot_dir / "catalog.sqlite")
    try:
        duplicate_groups = {row["evidence_id"]: row["duplicate_group_id"] for row in conn.execute(
            "SELECT evidence_id,duplicate_group_id FROM evidence")}
    finally:
        conn.close()
    known_ids = set(duplicate_groups)
    service = KnowledgeService(snapshots_root, cache_size=0)
    cases = []
    for case in labels["cases"]:
        relevant = case["relevant_evidence_ids"]
        if not relevant or not set(relevant) <= known_ids:
            raise ValueError("Every labeled case needs known, nonempty relevant evidence IDs")
        products = service.search_products(snapshot_id, case["filters"], limit=1)
        candidate_set = service.describe_candidate_set(snapshot_id, products["candidate_set_id"])
        if list(candidate_set["variant_ids"]) != [case["variant_id"]]:
            raise ValueError("Structural case does not select exactly its independently labeled variant")
        frozen_scope = _json(candidate_set)
        modes = {}
        for mode in ("keyword", "filtered"):
            result = service.retrieve_evidence(snapshot_id, case["query"], products["candidate_set_id"], k=5, mode=mode)
            ids = [item["evidence_id"] for item in result["evidence"]]
            valid = set(ids) & known_ids
            covered = _resolved_provenance(service, snapshot_id, sorted(valid), snapshot_dir)
            modes[mode] = {
                **retrieval_metrics(ids, relevant, duplicate_groups),
                "retrieved_evidence_ids": ids, "valid_evidence_count": len(valid),
                "citation_provenance_covered_count": len(covered), "retrieved_count": len(ids),
                "id_validity": len(valid) / len(ids) if ids else None,
                "citation_provenance_coverage": len(covered) / len(ids) if ids else None,
                "serialized_response_bytes": len(_json(result).encode("utf-8")),
            }
            if _json(service.describe_candidate_set(snapshot_id, products["candidate_set_id"])) != frozen_scope:
                raise RuntimeError("Candidate IDs or selected price IDs changed between modes")
        cases.append({"case_id": case["case_id"], "category": case["category"], "query": case["query"],
                      "candidate_set_id": products["candidate_set_id"],
                      "candidate_variant_ids": list(candidate_set["variant_ids"]),
                      "selected_price_record_ids": list(candidate_set["selected_price_record_ids"]),
                      "candidate_scope_sha256": sha256(frozen_scope.encode()).hexdigest(),
                      "same_candidate_and_price_scope": True,
                      "relevant_evidence_ids": relevant, "modes": modes})
    summaries = {}
    for mode in ("keyword", "filtered"):
        rows = [case["modes"][mode] for case in cases]
        total, valid, covered = (sum(row[name] for row in rows) for name in (
            "retrieved_count", "valid_evidence_count", "citation_provenance_covered_count"))
        summaries[mode] = {
            "recall_at_5": sum(row["recall_at_5"] for row in rows) / len(rows) if rows else None,
            "mrr_at_5": sum(row["mrr_at_5"] for row in rows) / len(rows) if rows else None,
            "retrieved_count": total, "id_validity": valid / total if total else None,
            "citation_provenance_coverage": covered / total if total else None,
        }
    unchanged = _tree_fingerprint(snapshot_dir) == before
    if not unchanged:
        raise RuntimeError("Immutable source snapshot changed during evaluation")
    return {
        "snapshot_id": snapshot_id, "label_version": LABEL_VERSION,
        "label_kind": labels["label_kind"], "split": "development", "case_count": len(cases),
        "labels_sha256": sha256(label_bytes).hexdigest(),
        "source_manifest_sha256": labels["snapshot_manifest_sha256"],
        "same_candidate_and_price_scope": True, "source_snapshot_unchanged": unchanged,
        "metric_scope": "group-aware, narrow explicit-attribute development labels; macro Recall@5/MRR@5",
        "citation_coverage_definition": "retrieved evidence IDs resolvable to existing, hash-matching archived JSON; not final-answer claim coverage",
        "modes": summaries, "cases": cases, "limitations": labels["limitations"],
    }
