"""Retrieval metrics (Task 6, docs/evaluation-metrics.md section 3): Recall@5 and MRR@5 against human labels.

Labels must not come only from what one retriever returned, so the pool to label is the union of the top 10
of both retrieval modes (keyword and filtered) over the same candidate set the single-call baselines use.
Reviewers fill `relevant` (true/false) in the pool file; unlabelled rows are ignored, and a case with no
relevant evidence is "not applicable" for Recall/MRR (never counted as 1).

  python -m evaluations.retrieval pool  --out evaluations/relevance_pool.jsonl
  python -m evaluations.retrieval score --labels evaluations/relevance_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODES = ("keyword", "filtered")


def build_pool(kb, snapshot_id: str, cases: list[dict], depth: int = 10) -> list[dict]:
    from .run import candidates_for
    rows = []
    for case in cases:
        _, set_id = candidates_for(kb, snapshot_id, case)
        seen = {}
        for mode in MODES:
            res = kb.retrieve_evidence(snapshot_id, case["retrieval_query"], set_id, k=depth, mode=mode)
            for rank, ev in enumerate(res["evidence"], 1):
                row = seen.setdefault(ev["evidence_id"], {
                    "case_id": case["case_id"], "split": case["split"], "query": case["retrieval_query"],
                    "evidence_id": ev["evidence_id"], "title": ev.get("title"),
                    "excerpt": (ev.get("text") or ev.get("excerpt") or "")[:400], "ranks": {}, "relevant": None})
                row["ranks"][mode] = rank
        rows += list(seen.values())
    return rows


def ranking(kb, snapshot_id: str, case: dict, mode: str, k: int = 5) -> list[str]:
    from .run import candidates_for
    _, set_id = candidates_for(kb, snapshot_id, case)
    return [e["evidence_id"] for e in kb.retrieve_evidence(snapshot_id, case["retrieval_query"], set_id, k=k, mode=mode)["evidence"]]


def recall_mrr(relevant: set[str], top5: list[str]):
    """(Recall@5, MRR@5) for one query, or None when there is no relevant evidence (not applicable)."""
    if not relevant:
        return None
    top = top5[:5]
    recall = len(relevant & set(top)) / len(relevant)
    rr = next((1 / (i + 1) for i, e in enumerate(top) if e in relevant), 0.0)
    return recall, rr


def score_rankings(labels: list[dict], rankings: dict[tuple[str, str], list[str]]) -> dict:
    """rankings[(case_id, mode)] = top-5 evidence IDs. Returns per-mode means over applicable labelled cases."""
    relevant, labelled = {}, set()
    for row in labels:
        if row.get("relevant") is None:
            continue
        labelled.add(row["case_id"])
        if row["relevant"]:
            relevant.setdefault(row["case_id"], set()).add(row["evidence_id"])
    out = {}
    for mode in MODES:
        per_case = {}
        for case_id in sorted(labelled):
            r = recall_mrr(relevant.get(case_id, set()), rankings.get((case_id, mode), []))
            if r is not None:
                per_case[case_id] = r
        out[mode] = {
            "recall_at_5": round(statistics.mean(v[0] for v in per_case.values()), 3) if per_case else None,
            "mrr_at_5": round(statistics.mean(v[1] for v in per_case.values()), 3) if per_case else None,
            "applicable_cases": len(per_case), "labelled_cases": len(labelled),
            "not_applicable": len(labelled) - len(per_case)}
    return out


def _kb(config: Path, root: Path):
    from knowledge.bootstrap import install_snapshot
    from knowledge.service import KnowledgeService
    from knowledge.settings import load_settings
    s = load_settings(config, data_root=root, snapshots_root=root / "snapshots")
    install_snapshot(s)
    return KnowledgeService(s.snapshots_root, **s.service), s.snapshot_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pool", "score"])
    here = Path(__file__).resolve().parent
    ap.add_argument("--cases", type=Path, default=here / "cases.jsonl")
    ap.add_argument("--config", type=Path, default=here.parent / "configs/knowledge.json")
    ap.add_argument("--data-root", type=Path, required=True, help="where to install/read the snapshot")
    ap.add_argument("--out", type=Path, default=here / "relevance_pool.jsonl")
    ap.add_argument("--labels", type=Path, default=here / "relevance_pool.jsonl")
    a = ap.parse_args()
    cases = [json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    kb, sid = _kb(a.config, a.data_root)
    if a.action == "pool":
        rows = build_pool(kb, sid, cases)
        a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} pooled evidence rows for {len(cases)} cases -> {a.out}")
    else:
        labels = [json.loads(l) for l in a.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {c["case_id"]: c for c in cases}
        rankings = {(cid, m): ranking(kb, sid, by_id[cid], m) for cid in {r["case_id"] for r in labels} for m in MODES}
        print(json.dumps(score_rankings(labels, rankings), indent=2))


if __name__ == "__main__":
    main()
