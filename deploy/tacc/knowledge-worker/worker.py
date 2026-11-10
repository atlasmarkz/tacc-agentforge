"""Validate/query a staged immutable snapshot on a TACC CPU worker."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from knowledge.bootstrap import install_snapshot
from knowledge.service import KnowledgeService
from knowledge.settings import load_settings
from knowledge.store import verify_snapshot


def run():
    workdir = Path(os.environ["TACC_WORKDIR"]).resolve()
    userdir = Path(os.environ["TACC_USERDIR"]).resolve()
    config_path = Path(os.environ.get("KNOWLEDGE_RUNTIME_CONFIG", workdir / "runtime.json"))
    if not config_path.is_file():
        if os.environ.get("KNOWLEDGE_RUNTIME_CONFIG"):
            raise ValueError("Configured KNOWLEDGE_RUNTIME_CONFIG does not exist")
        config_path = workdir / "runtime.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"mode", "data_subdir", "copy_snapshot_to_local", "benchmark_iterations",
               "benchmark_scale", "query", "category", "snapshot_id"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Invalid knowledge-worker runtime settings")
    mode = config.get("mode", "smoke")
    if not isinstance(mode, str) or mode not in {"verify", "smoke", "benchmark"}:
        raise ValueError("Worker mode must be verify, smoke or benchmark")
    subdir = Path(config.get("data_subdir", "tacc-agentforge-data"))
    if subdir.is_absolute() or ".." in subdir.parts or not subdir.parts:
        raise ValueError("data_subdir must stay inside TACC_USERDIR")
    if type(config.get("copy_snapshot_to_local", True)) is not bool:
        raise ValueError("copy_snapshot_to_local must be a boolean")
    settings = load_settings(workdir / "_stage/pkg/configs/knowledge.json",
                             data_root=userdir / subdir,
                             snapshots_root=userdir / subdir / "data/knowledge/snapshots",
                             snapshot_id=config.get("snapshot_id"))
    installed = install_snapshot(settings)
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job identifier")
    output_dir = userdir / "tacc-agentforge-results" / ("knowledge-" + job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        query_root = settings.snapshots_root
        if config.get("copy_snapshot_to_local", True):
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(
                prefix="knowledge-query-", dir=os.environ.get("SLURM_TMPDIR"))))
            query_root = scratch / "snapshots"
            shutil.copytree(settings.snapshots_root / settings.snapshot_id, query_root / settings.snapshot_id)
        query_settings = replace(settings, snapshots_root=query_root)
        verification = verify_snapshot(query_root / settings.snapshot_id)
        if not verification["valid"]:
            raise ValueError("Query snapshot failed verification")
        report = {"status": "ready", "mode": mode, "job_id": job_id,
                  "execution_context": "local_simulation" if os.environ.get("KNOWLEDGE_LOCAL_TEST") == "1" else "tacc_job",
                  "python": sys.version.split()[0], "snapshot_id": settings.snapshot_id,
                  "persistent_snapshot_path": installed["snapshot_path"],
                  "query_snapshot_path": str(query_root / settings.snapshot_id),
                  "copied_to_node_scratch": config.get("copy_snapshot_to_local", True),
                  "bootstrap_status": installed["status"], "verification": verification,
                  "external_calls": 0, "gpu_required": False}
        if mode == "smoke":
            service = KnowledgeService(query_settings.snapshots_root, **query_settings.service)
            category = config.get("category", "earbuds")
            page = service.search_products(settings.snapshot_id, {"category": category}, limit=3)
            evidence = service.retrieve_evidence(settings.snapshot_id, config.get("query", "noise cancelling earbuds"),
                                                 page["candidate_set_id"], k=5)
            resolved = service.resolve_evidence(settings.snapshot_id,
                                                [row["evidence_id"] for row in evidence["evidence"][:2]])
            report.update(products=page, retrieval=evidence, resolved=resolved)
        elif mode == "benchmark":
            from knowledge.benchmark import benchmark
            report["benchmark"] = benchmark(query_root, settings.snapshot_id,
                iterations=config.get("benchmark_iterations", 100), scale=config.get("benchmark_scale", 10))
        output = output_dir / (mode + ".json")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {"products", "retrieval", "resolved", "benchmark"}},ensure_ascii=False))
        print("KB_REPORT=" + str(output))
    return report


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, KeyError) as exc:
        print("Knowledge worker failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
