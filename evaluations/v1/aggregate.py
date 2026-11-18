"""v1-specific aggregation.

This module is deliberately independent of `evaluations/score.py`: the v0 scorer
and its frozen reports must stay byte-compatible with `evaluations/freeze.json`.
The v1 benchmark uses the stricter BVR denominator (known total only), the
`insufficient_evidence` stratum and explicit invalid-action counting.
"""
from __future__ import annotations

import random
import statistics


def _pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 3)


def case_bootstrap(per_case: dict[str, list[float]], n: int = 2000, seed: int = 0):
    """Mean success with a 95% interval, resampling cases (repeats of one case stay together)."""
    cases = sorted(per_case)
    if not cases:
        return None
    means = {c: statistics.mean(per_case[c]) for c in cases}
    point = statistics.mean(means.values())
    rng = random.Random(seed)
    draws = sorted(statistics.mean(means[rng.choice(cases)] for _ in cases) for _ in range(n))
    return {"mean": round(point, 3), "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n) - 1], 3)],
            "cases": len(cases)}


def aggregate(scores: list[dict], records: list[dict]) -> dict:
    """Per-configuration summary. Denominators are reported with every rate."""
    by_config = {}
    rec_index = {(r["config"], r["case_id"], r["rep"]): r for r in records}
    for s in scores:
        by_config.setdefault(s["config"], []).append(s)
    report = {}
    for config, ss in sorted(by_config.items()):
        per_case = {}
        for s in ss:
            per_case.setdefault(s["case_id"], []).append(1.0 if s["success"] else 0.0)
        plans = [s for s in ss if s["plan_delivered"]]
        known_total = [s for s in plans if s["plan_total"] is not None]
        recs = [rec_index[(s["config"], s["case_id"], s["rep"])] for s in ss]
        walls = [r.get("wall_s") for r in recs if r.get("wall_s") is not None]

        def rate(sub, key):
            return {"n": sum(1 for s in sub if s.get(key)), "m": len(sub),
                    "rate": round(sum(1 for s in sub if s.get(key)) / len(sub), 3) if sub else None}

        by_expected = {}
        for exp in ("plan", "no_feasible_plan", "clarify", "insufficient_evidence"):
            sub = [s for s in ss if s["expected"] == exp]
            if sub:
                by_expected[exp] = rate(sub, "success")
        by_category = {}
        for cat in sorted({s["category"] for s in ss}):
            by_category[cat] = rate([s for s in ss if s["category"] == cat], "success")
        reasons = {}
        for s in ss:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        report[config] = {
            "runs": len(ss), "tsr": rate(ss, "success"), "tsr_by_case": case_bootstrap(per_case),
            "tsr_by_expected": by_expected, "tsr_by_category": by_category,
            "bvr": {"n": sum(1 for s in known_total if s["over_budget"]), "m": len(known_total),
                    "rate": (round(sum(1 for s in known_total if s["over_budget"]) / len(known_total), 3)
                             if known_total else None)},
            "plans_delivered": len(plans),
            "constraint_violation": rate(plans, "violates"), "unverifiable_claim": rate(plans, "unverifiable"),
            "ungrounded_ids": rate(plans, "ungrounded"),
            "evidence_of_another_product": rate(plans, "evidence_mismatch"),
            "ids_not_in_context": rate([s for s in plans if s.get("context_checked")], "ids_not_in_context"),
            "amount_error_rate": None,
            "outcome_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "wall_s": {"p50": _pct(walls, 0.5), "p95": _pct(walls, 0.95), "max": _pct(walls, 1.0)},
            "model_calls_mean": _mean(r.get("model_calls") for r in recs),
            "prompt_tokens_mean": _mean(r.get("prompt_tokens") for r in recs),
            "completion_tokens_mean": _mean(r.get("completion_tokens") for r in recs),
            "invalid_actions": sum(max(int(r.get("invalid_actions") or 0),
                                       1 if r.get("stop_reason") == "invalid_action" else 0) for r in recs),
        }
    return report


def paired_difference(scores: list[dict], a: str, b: str, n: int = 2000, seed: int = 0):
    """TSR(b) - TSR(a) over the cases both configurations ran, with a case-level bootstrap interval."""
    def per_case(cfg):
        out = {}
        for s in scores:
            if s["config"] == cfg:
                out.setdefault(s["case_id"], []).append(1.0 if s["success"] else 0.0)
        return {c: statistics.mean(v) for c, v in out.items()}
    pa, pb = per_case(a), per_case(b)
    common = sorted(set(pa) & set(pb))
    if not common:
        return None
    diffs = {c: pb[c] - pa[c] for c in common}
    rng = random.Random(seed)
    draws = sorted(statistics.mean(diffs[rng.choice(common)] for _ in common) for _ in range(n))
    return {"from": a, "to": b, "diff": round(statistics.mean(diffs.values()), 3),
            "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n) - 1], 3)], "cases": len(common)}


def _mean(values):
    v = [x for x in values if x is not None]
    return round(statistics.mean(v), 1) if v else None
