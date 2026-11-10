#!/usr/bin/env python3
"""Stage only public KB code/configs and the pinned seed for SDK upload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from knowledge.settings import load_settings


def stage(dry_run=False):
    settings = load_settings(ROOT / "configs/knowledge.json")
    if hashlib.sha256(settings.seed_archive.read_bytes()).hexdigest() != settings.seed_sha256:
        raise ValueError("Pinned snapshot archive failed SHA256 check")
    summary = {"snapshot_id": settings.snapshot_id, "seed_sha256": settings.seed_sha256,
               "stage_path": str(HERE / "_stage"), "external_calls": 0, "dry_run": dry_run}
    if dry_run:
        return summary
    destination = HERE / "_stage"
    if destination.exists() and not (destination / ".knowledge-stage").is_file():
        raise ValueError("Refusing to replace a staging directory without its ownership marker")
    temporary = Path(tempfile.mkdtemp(prefix=".stage-build-", dir=HERE))
    try:
        pkg = temporary / "pkg"
        shutil.copytree(ROOT / "knowledge", pkg / "knowledge",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        (pkg / "configs").mkdir()
        for name in ("knowledge.json", "category_schemas.json", "shopping_queries.json"):
            shutil.copy2(ROOT / "configs" / name, pkg / "configs" / name)
        # Preserve the archive's configured repository-relative location.
        relative_seed = settings.seed_archive.relative_to(ROOT)
        target = pkg / relative_seed
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.seed_archive, target)
        (temporary / ".knowledge-stage").write_text("generated knowledge-worker upload bundle\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    runtime = HERE / "runtime.json"
    if not runtime.exists():
        shutil.copy2(HERE / "runtime.example.json", runtime)
    summary["files_staged"] = sum(p.is_file() for p in destination.rglob("*"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = stage(args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
