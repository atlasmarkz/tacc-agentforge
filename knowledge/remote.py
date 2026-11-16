"""Download a pinned snapshot archive from object storage (issue #25).

Snapshots are published once per ID under `gs://<bucket>/snapshots/<snapshot_id>.zip` and never overwritten
(deploy/gcp/publish-snapshot.sh). The archive is the same ZIP that `bootstrap` already verifies, so a download
only replaces "read the ZIP from the repository": the pinned SHA256, the entry checks and the manifest
verification in `install_snapshot` stay exactly as they are.

Supported URLs:
  gs://bucket/path.zip     via `gcloud storage cp` (on GCE the VM's service account is used; elsewhere the
                           caller's `gcloud auth` login)
  https://host/path.zip    plain HTTPS (e.g. a signed URL or a GitHub release asset)
  file:///path/to.zip      local file (tests, mirrors)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_archive(url: str, directory: Path, sha256: str, *, timeout: float = 300) -> Path:
    """Download `url` into `directory` and return the file; raises ValueError unless its SHA256 is `sha256`.

    The file is written under a temporary name and only renamed once the hash matches, so a partial or wrong
    download never looks like a usable archive.
    """
    scheme = urlsplit(url).scheme
    if scheme not in ("gs", "https", "file"):
        raise ValueError("Snapshot URL must be gs://, https:// or file://")
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".zip", dir=directory)
    os.close(fd)
    partial = Path(name)
    try:
        if scheme == "gs":
            gcloud = shutil.which("gcloud")
            if gcloud is None:
                raise ValueError("gcloud is required to download gs:// snapshots")
            done = subprocess.run([gcloud, "storage", "cp", url, str(partial)], capture_output=True, text=True,
                                  timeout=timeout)
            if done.returncode != 0:
                raise ValueError(f"gcloud storage cp failed: {done.stderr.strip()[-300:]}")
        else:
            with urlopen(url, timeout=timeout) as response, partial.open("wb") as out:  # noqa: S310 (scheme checked)
                total = 0
                for block in iter(lambda: response.read(1 << 20), b""):
                    total += len(block)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Snapshot archive exceeds the size limit")
                    out.write(block)
        if partial.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Snapshot archive exceeds the size limit")
        if _sha256(partial) != sha256:
            raise ValueError("Downloaded snapshot archive SHA256 mismatch")
        final = directory / f"{sha256}.zip"
        os.replace(partial, final)
        return final
    finally:
        partial.unlink(missing_ok=True)
