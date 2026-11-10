"""Install a pinned, verified seed snapshot without any network or API key."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import sys
import tempfile
import zipfile

from .settings import KnowledgeSettings
from .store import verify_snapshot


def install_snapshot(settings: KnowledgeSettings, *, archive=None, sha256=None):
    destination = settings.snapshots_root / settings.snapshot_id
    if destination.exists():
        existing = verify_snapshot(destination)
        if not existing["valid"]:
            raise ValueError("Existing snapshot is invalid; refusing to replace it")
        if existing["snapshot_id"] != settings.snapshot_id:
            raise ValueError("Existing snapshot ID does not match configuration")
        return {"status": "already_installed", "snapshot_id": settings.snapshot_id,
                "snapshot_path": str(destination), "verification": existing}
    source = Path(archive).expanduser().resolve() if archive is not None else settings.seed_archive
    expected = sha256 or settings.seed_sha256
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Seed archive SHA256 mismatch")
    settings.snapshots_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bootstrap-", dir=settings.snapshots_root))
    try:
        seen, total = set(), 0
        with zipfile.ZipFile(source) as package:
            if len(package.infolist()) > 20000:
                raise ValueError("Seed archive has too many entries")
            for item in package.infolist():
                name = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if ("\\" in item.filename or name.is_absolute() or ".." in name.parts or
                    not name.parts or name.parts[0] != settings.snapshot_id or
                    stat.S_ISLNK(mode) or item.flag_bits & 1):
                    raise ValueError("Unsafe seed archive entry")
                if item.filename in seen:
                    raise ValueError("Duplicate seed archive entry")
                seen.add(item.filename)
                total += item.file_size
                if total > 512 * 1024 * 1024:
                    raise ValueError("Seed archive exceeds the size limit")
                target = temporary.joinpath(*name.parts)
                if not target.resolve().is_relative_to(temporary):
                    raise ValueError("Seed archive path escapes destination")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(item) as reader, target.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
        extracted = temporary / settings.snapshot_id
        verification = verify_snapshot(extracted)
        if not verification["valid"] or verification.get("snapshot_id") != settings.snapshot_id:
            raise ValueError("Extracted seed snapshot failed verification")
        try:
            extracted.rename(destination)
        except OSError:
            # Concurrent first-use installs may finish first; never overwrite them.
            existing = verify_snapshot(destination) if destination.is_dir() else {"valid": False}
            if not existing["valid"] or existing.get("snapshot_id") != settings.snapshot_id:
                raise
        return {"status": "installed", "snapshot_id": settings.snapshot_id,
                "snapshot_path": str(destination), "archive_sha256": expected,
                "verification": verification}
    except zipfile.BadZipFile as exc:
        raise ValueError("Seed archive is not a valid ZIP") from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def diagnose(settings: KnowledgeSettings):
    supported = sys.version_info >= (3, 12)
    connection = sqlite3.connect(":memory:")
    try:
        try:
            connection.execute("CREATE VIRTUAL TABLE capability USING fts5(text)")
            fts = True
        except sqlite3.Error:
            fts = False
    finally:
        connection.close()
    verification = verify_snapshot(settings.snapshots_root / settings.snapshot_id)
    return {"ready": supported and fts and verification["valid"],
            "python": sys.version.split()[0], "python_supported": supported,
            "sqlite": sqlite3.sqlite_version, "fts5": fts,
            "config_path": str(settings.config_path), "data_root": str(settings.data_root),
            "snapshots_root": str(settings.snapshots_root), "snapshot_id": settings.snapshot_id,
            "service": settings.service, "verification": verification}
