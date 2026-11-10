"""Manage auditable collection, knowledge snapshots, queries and local evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]


def _write_result(result, output=None):
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--settings", type=Path, help="Knowledge configuration JSON")
    parser.add_argument("--config", type=Path, default=REPO / "configs/shopping_queries.json")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("bootstrap", help="Install the pinned offline seed snapshot")
    p.add_argument("--archive", type=Path)
    p.add_argument("--sha256")
    p.add_argument("--snapshot-id")
    p.add_argument("--snapshots-root", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("doctor", help="Check Python, FTS5 and the selected snapshot")
    p.add_argument("--snapshot-id")
    p.add_argument("--snapshots-root", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("plan", help="Plan new detail targets offline")
    p.add_argument("--target-per-category", type=int, default=10)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("collect", help="Explicit quota-bounded incremental acquisition")
    p.add_argument("--session-dir", type=Path)
    p.add_argument("--key-file", type=Path)
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--target-per-category", type=int, default=10)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("ingest", help="Idempotently import verified acquisition runs")
    p.add_argument("--run-dir", action="append", type=Path)
    p.add_argument("--db", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("snapshot", help="Export a frozen, portable snapshot")
    p.add_argument("--db", type=Path)
    p.add_argument("--snapshots-root", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("verify", help="Validate every manifest hash and evidence pointer")
    p.add_argument("snapshot_dir", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("audit", help="Inspect data coverage, conflicts and missing facts")
    p.add_argument("--db", type=Path)
    p.add_argument("--output", type=Path)
    for name in ("products", "search", "offers", "evidence", "benchmark", "evaluate", "label-development"):
        p = commands.add_parser(name)
        p.add_argument("--snapshot-id", help="Override the configured immutable snapshot ID")
        p.add_argument("--snapshots-root", type=Path)
        p.add_argument("--output", type=Path)
        if name in ("products", "search"):
            p.add_argument("--category")
            p.add_argument("--currency")
            p.add_argument("--max-price-minor", type=int)
            p.add_argument("--filters", default="{}", help="Additional JSON structured filters")
        if name == "products":
            p.add_argument("--limit", type=int, default=5)
            p.add_argument("--cursor")
        elif name == "search":
            p.add_argument("query")
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--mode", choices=("filtered", "keyword"), default="filtered")
        elif name == "offers":
            p.add_argument("offer_ids", nargs="+")
        elif name == "evidence":
            p.add_argument("evidence_ids", nargs="+")
            p.add_argument("--cursor")
        elif name == "benchmark":
            p.add_argument("--iterations", type=int, default=100)
            p.add_argument("--scale", type=int, default=10)
        elif name == "evaluate":
            p.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from .settings import load_settings
        settings = load_settings(args.settings, data_root=args.data_root,
            snapshots_root=getattr(args,"snapshots_root",None), snapshot_id=getattr(args,"snapshot_id",None))
        data_root = settings.data_root
        if hasattr(args,"snapshot_id"):
            args.snapshot_id = settings.snapshot_id
        db = getattr(args, "db", None) or data_root / "data/knowledge/catalog.sqlite"
        snapshots_root = settings.snapshots_root
        if args.command == "bootstrap":
            from .bootstrap import install_snapshot
            result = install_snapshot(settings,archive=args.archive,sha256=args.sha256)
        elif args.command == "doctor":
            from .bootstrap import diagnose
            result = diagnose(settings)
        elif args.command == "plan":
            from .acquisition import plan_collection
            result = plan_collection(data_root, args.config, args.target_per_category)
        elif args.command == "collect":
            from .acquisition import collect_incremental
            from .credentials import load_api_key
            session = args.session_dir or data_root / "data/serpapi" / (
                "run-local-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
            key = load_api_key(args.key_file)
            result = collect_incremental(data_root, args.config, session, key,
                                         args.max_attempts, args.target_per_category)
        elif args.command == "ingest":
            from .store import ingest_runs
            runs = args.run_dir or sorted((data_root / "data/serpapi").glob("run-*"))
            result = ingest_runs(runs, db, args.config, REPO / "configs/category_schemas.json")
        elif args.command == "snapshot":
            from .store import export_snapshot
            result = export_snapshot(db, snapshots_root)
        elif args.command == "verify":
            from .store import verify_snapshot
            result = verify_snapshot(args.snapshot_dir)
        elif args.command == "audit":
            from .store import audit_database
            if args.db is None and not db.exists():
                db = snapshots_root / settings.snapshot_id / "catalog.sqlite"
            result = audit_database(db)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(snapshots_root, args.snapshot_id, args.iterations, args.scale)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(snapshots_root, args.snapshot_id, args.labels)
        elif args.command == "label-development":
            from .evaluation import prepare_development_labels
            if args.output is None:
                raise ValueError("label-development requires --output outside the immutable snapshot")
            result = prepare_development_labels(snapshots_root / args.snapshot_id,args.output)
            # The label writer refuses replacement and freezes its own document.
            # Its return value is a summary, not the label document to overwrite.
            _write_result(result)
            return 0
        else:
            from .service import KnowledgeService
            service = KnowledgeService(snapshots_root,**settings.service)
            try:
                if args.command in ("products", "search"):
                    filters = json.loads(args.filters)
                    if not isinstance(filters, dict):
                        raise ValueError("--filters must be a JSON object")
                    for name in ("category", "currency", "max_price_minor"):
                        value = getattr(args, name)
                        if value is not None:
                            filters[name] = value
                    result = service.search_products(args.snapshot_id, filters,
                                                     limit=getattr(args, "limit", 5),
                                                     cursor=getattr(args, "cursor", None))
                    if args.command == "search":
                        result = service.retrieve_evidence(args.snapshot_id,args.query,
                            result["candidate_set_id"],args.k,mode=args.mode)
                elif args.command == "offers":
                    result = service.get_offer_facts(args.snapshot_id,args.offer_ids)
                else:
                    result = service.resolve_evidence(args.snapshot_id,args.evidence_ids,args.cursor)
            finally:
                close = getattr(service,"close",None)
                if close:
                    close()
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        print(json.dumps({"error":getattr(exc,"code","operation_failed"),
                          "message":getattr(exc,"public_message",str(exc))},ensure_ascii=False),file=sys.stderr)
        return 2
    _write_result(result,getattr(args,"output",None))
    return 1 if (args.command == "verify" and not result.get("valid")) or (args.command == "doctor" and not result.get("ready")) else 0
