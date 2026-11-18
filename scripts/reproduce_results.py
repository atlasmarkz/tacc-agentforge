"""Recompute the reported results from the committed raw data, without TACC or a GPU.

    python scripts/reproduce_results.py [--out DIR]

1. Agent baseline (docs/experiments/baseline.md, Section 4): re-scores every recorded attempt in
   evaluations/results/frozen-heldout/ and frozen-dev/ against the frozen catalog snapshot
   (assets/knowledge/pilot-snapshot.zip) with the independent enumerator, and prints the task success
   rate (TSR), budget violation rate (BVR) and paired differences per configuration.
2. Serving experiments (inference/RESULTS.md): regenerates the report from inference/results/.

Each part is compared with the committed copy and printed as MATCH or DIFFERS. Exit code 0 only if all match.
This checks the scoring and aggregation; producing new raw records needs the model on TACC
(see "Detailed Instructions" in README.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluations.run import rescore  # noqa: E402
from inference import report as inference_report  # noqa: E402

SNAPSHOT_ZIP = ROOT / "assets/knowledge/pilot-snapshot.zip"
SNAPSHOT_ID = "snapshot_5bcd00723fb0b7589cbe50c8"
BASELINES = ("frozen-heldout", "frozen-dev")
SERVING_FOLDERS = ("serving27b-*", "graphs-*")


def frac(x: dict) -> str:
    return f"{x['n']}/{x['m']}"


def check_baseline(name: str, snapshot_db: Path, out: Path) -> bool:
    src = ROOT / "evaluations/results" / name
    dst = out / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "records.jsonl", dst / "records.jsonl")
    report = rescore(dst, ROOT / "evaluations/cases.jsonl", snapshot_db)
    committed = json.loads((src / "report.json").read_text(encoding="utf-8"))

    print(f"\n== {name}: {report['cases']} cases x {report['repeats_complete']} repeats "
          f"({report['records_total']} attempts)")
    print(f"{'config':<7} {'TSR':>7}  {'per-case mean (95% CI)':<24} {'BVR':>6}  plans")
    for config, s in report["summary"].items():
        ci = s["tsr_by_case"]["ci95"]
        print(f"{config:<7} {frac(s['tsr']):>7}  {s['tsr_by_case']['mean']:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
              f"{'':<8} {frac(s['bvr']):>6}  {s['plans_delivered']}")
    for p in report["paired"]:
        print(f"  paired {p['from']} -> {p['to']}: {p['diff']:+.2f} (95% CI {p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f})")

    same = report["summary"] == committed["summary"] and report["paired"] == committed.get("paired")
    print(f"{'MATCH' if same else 'DIFFERS'}: {name} vs evaluations/results/{name}/report.json")
    return same


def check_serving(out: Path) -> bool:
    folders = sorted(p for pattern in SERVING_FOLDERS for p in (ROOT / "inference/results").glob(pattern))
    target = out / "inference-RESULTS.md"
    argv = sys.argv
    try:
        sys.argv = ["report", *map(str, folders), "--out", str(target),
                    "--notes", str(ROOT / "inference/results/notes.md")]
        inference_report.main()
    finally:
        sys.argv = argv
    same = target.read_text(encoding="utf-8") == (ROOT / "inference/RESULTS.md").read_text(encoding="utf-8")
    print(f"\n== serving experiments: {len(folders)} result folders")
    print(f"{'MATCH' if same else 'DIFFERS'}: {target} vs inference/RESULTS.md")
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="output directory (default: a new temporary directory)")
    a = ap.parse_args()
    out = (a.out or Path(tempfile.mkdtemp(prefix="agentforge-reproduce-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(SNAPSHOT_ZIP) as z:
        z.extract(f"{SNAPSHOT_ID}/catalog.sqlite", out / "snapshot")
    snapshot_db = out / "snapshot" / SNAPSHOT_ID / "catalog.sqlite"

    results = [check_baseline(name, snapshot_db, out) for name in BASELINES]
    results.append(check_serving(out))
    print(f"\nOutputs in {out}")
    print("ALL MATCH" if all(results) else "SOME RESULTS DIFFER")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
