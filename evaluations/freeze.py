"""Freeze manifest for the Task 6 baseline: what "the system" is when the held-out split is run.

The manifest (evaluations/freeze.json) records a SHA-256 for every file that defines the evaluated system and
its measurement: harness code and prompts, knowledge tools, configuration (policy profile, snapshot ID),
model adapter and client, the vLLM serving arguments, the cases and the scorer. It also records the resolved
harness policy and model settings, so a report says exactly what ran.

    python -m evaluations.freeze write     # freeze the current tree (then commit freeze.json)
    python -m evaluations.freeze check     # list what changed since the freeze; exit 1 if anything did

run.py refuses --split heldout unless the tree it runs from matches the manifest. Line endings are normalised
to LF before hashing, so a Windows checkout and the staged Linux copy agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("freeze.json")
PROFILE = "tacc-fast"            # what app/harness_service.py serves by default (thinking off)

# group -> glob patterns relative to the repository root (or the staged package root on TACC)
GROUPS = {
    "harness": ["agent/*.py", "agent/requirements.lock"],
    "knowledge_tools": ["knowledge/*.py", "knowledge/schema.sql"],
    "configs": ["configs/agent.yaml", "configs/knowledge.json", "configs/category_schemas.json"],
    "model_adapter": ["inference/client.py", "inference/model_port.py"],
    "serving": ["deploy/tacc/service/qwen-service/service.json", "deploy/tacc/service/qwen-service/serve.py"],
    "evaluation": ["evaluations/run.py", "evaluations/score.py", "evaluations/catalog.py",
                   "evaluations/cases.jsonl"],
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def collect(root: Path) -> dict[str, dict[str, str]]:
    files = {}
    for group, patterns in GROUPS.items():
        found = {}
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                if p.is_file():
                    found[p.relative_to(root).as_posix()] = file_hash(p)
        files[group] = found
    return files


def resolved_settings(root: Path) -> dict:
    """The policy and model settings the harness configurations run with (same loader as the deployed service)."""
    from agent.settings import load_runtime_settings
    s = load_runtime_settings(root / "configs/agent.yaml", PROFILE)
    model = s.model.model_dump(mode="json")
    return {"profile": PROFILE, "policy": s.policy.model_dump(mode="json"),
            "model": {k: model[k] for k in ("name", "temperature", "enable_thinking", "constrain_output",
                                            "timeout_margin_s", "max_attempts")}}


def case_summary(root: Path) -> dict:
    lines = [l for l in (root / "evaluations/cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [json.loads(l) for l in lines]
    out = {}
    for split in sorted({c["split"] for c in cases}):
        ids = sorted(c["case_id"] for c in cases if c["split"] == split)
        out[split] = {"cases": len(ids), "categories": sorted({c["category"] for c in cases if c["split"] == split}),
                      "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    return out


def build(root: Path = ROOT) -> dict:
    knowledge = json.loads((root / "configs/knowledge.json").read_text(encoding="utf-8"))
    service = json.loads((root / "deploy/tacc/service/qwen-service/service.json").read_text(encoding="utf-8"))
    files = collect(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema": 1, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit_at_freeze": commit,
            "digest": digest, "snapshot_id": knowledge.get("snapshot_id"),
            "serving": {k: service[k] for k in ("served_name", "tensor_parallel_size", "args") if k in service},
            "settings": resolved_settings(root), "cases": case_summary(root), "files": files}


def compare(manifest: dict, root: Path = ROOT, require_all: bool = True) -> list[str]:
    """Differences between the manifest and the tree at root. With require_all=False, files that are absent
    here (not staged) are skipped rather than reported."""
    current = collect(root)
    problems = []
    for group, frozen in manifest["files"].items():
        now = current.get(group, {})
        for path, h in frozen.items():
            if path not in now:
                if require_all:
                    problems.append(f"missing {path}")
            elif now[path] != h:
                problems.append(f"changed {path}")
        problems += [f"added {path}" for path in now if path not in frozen]
    settings = resolved_settings(root)
    if settings != manifest["settings"]:
        problems.append("resolved policy or model settings differ")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["write", "check", "show"])
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.action == "write":
        m = build(a.root)
        MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"frozen {sum(len(v) for v in m['files'].values())} files, digest {m['digest'][:16]}, "
              f"snapshot {m['snapshot_id']} -> {MANIFEST}")
        return 0
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if a.action == "show":
        print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1, ensure_ascii=False))
        return 0
    problems = compare(m, a.root)
    for p in problems:
        print(p)
    print("matches the freeze" if not problems else f"{len(problems)} difference(s) from the freeze")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
