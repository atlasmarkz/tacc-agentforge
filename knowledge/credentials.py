"""Read acquisition credentials into memory without logging or modifying the source."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


def load_api_key(path: str | Path | None = None) -> str:
    value = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not value:
        if path is None:
            raise ValueError("Configure SERPAPI_API_KEY or provide --key-file")
        source = Path(path).resolve()
        if source.suffix.casefold() == ".rtf":
            try:
                result = subprocess.run(
                    ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(source)],
                    capture_output=True, check=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("Cannot decode the configured RTF credential file") from None
            value = result.stdout.decode("utf-8").strip()
        else:
            value = source.read_text(encoding="utf-8").strip()
    matches = re.findall(r"(?<![a-fA-F0-9])[a-fA-F0-9]{64}(?![a-fA-F0-9])", value)
    if len(matches) == 1:
        return matches[0]
    if re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value):
        return value
    raise ValueError("Credential format is ambiguous; expected exactly one API key")
