"""Portable host configuration; paths are relative to the repository, not cwd."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class KnowledgeSettings:
    config_path: Path
    project_root: Path
    data_root: Path
    snapshots_root: Path
    snapshot_id: str
    seed_archive: Path
    seed_sha256: str
    service: dict


def _path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Knowledge paths must be nonempty strings")
    expanded = os.path.expandvars(str(value))
    if re.search(r"\$[A-Za-z_{]", expanded):
        raise ValueError("Knowledge path contains an undefined environment variable")
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(config_path=None, *, data_root=None, snapshots_root=None, snapshot_id=None):
    """Explicit arguments > KNOWLEDGE_* env > versioned JSON defaults.

    Configuration paths are resolved against the directory above configs/.
    Environment and explicit relative overrides are resolved against current cwd.
    """
    path = Path(config_path or os.environ.get("KNOWLEDGE_CONFIG", REPO / "configs/knowledge.json")).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported knowledge configuration version")
    root = path.parent.parent
    selected_data = data_root if data_root is not None else os.environ.get("KNOWLEDGE_DATA_ROOT")
    data = _path(selected_data, Path.cwd()) if selected_data is not None else _path(value["data_root"], root)
    selected_snapshots = snapshots_root if snapshots_root is not None else os.environ.get("KNOWLEDGE_SNAPSHOTS_ROOT")
    snapshots = _path(selected_snapshots, Path.cwd()) if selected_snapshots is not None else data / "data/knowledge/snapshots"
    identifier = snapshot_id if snapshot_id is not None else os.environ.get("KNOWLEDGE_SNAPSHOT_ID", value.get("snapshot_id"))
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", identifier):
        raise ValueError("snapshot_id must be an identifier, not a filesystem path")
    seed = value.get("seed", {})
    expected = seed.get("sha256", "")
    if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
        raise ValueError("Seed archive needs a pinned SHA256")
    service = {"cache_size": 512, "max_result_bytes": 8192, "price_ttl_hours": 24}
    configured = value.get("service", {})
    if not isinstance(configured, dict) or set(configured) - set(service):
        raise ValueError("Unsupported knowledge service settings")
    service.update(configured)
    if type(service["cache_size"]) is not int or service["cache_size"] < 0:
        raise ValueError("cache_size must be a nonnegative integer")
    if type(service["max_result_bytes"]) is not int or not 1024 <= service["max_result_bytes"] <= 65536:
        raise ValueError("max_result_bytes must be between 1024 and 65536")
    if type(service["price_ttl_hours"]) not in (int, float) or not 0 < service["price_ttl_hours"] < float("inf"):
        raise ValueError("price_ttl_hours must be positive and finite")
    return KnowledgeSettings(path, root, data, snapshots, identifier,
                             _path(seed["archive"], root), expected, service)
