"""Deterministic scoring of evaluation runs (Task 6), following docs/evaluation-metrics.md section 5.

Each run record is one attempt at one case by one configuration. Its outcome is one of
  {"kind": "plan", "items": [{"offer_id", "quantity", "evidence_ids"}...]}
  {"kind": "ask_user", "question": "...", "required_fields": [...]}
  {"kind": "error", "error": "..."}
and is judged against the case's ground truth from the independent enumerator (evaluations/catalog.py).

Success (TSR numerator), by expected outcome:
- plan:             a plan whose every item is feasible (exists, right category, requirements met, cost known
                    and within budget on the case's basis) with the requested quantity; asking instead fails
- no_feasible_plan: no plan delivered (an explanation or question); any plan fails
- clarify:          a question that asks for the budget; a plan fails
Errors, timeouts and limits always fail. Text quality is not judged here (needs reviewers).

BVR = delivered plans whose known total exceeds the budget / delivered plans (n/m, None when m = 0).
Amount error rate needs amounts reported by the system; no configuration reports them yet, so it is None.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from .catalog import judge_plan

BUDGET_FIELD_HINTS = ("budget", "cap", "price", "spend", "预算")


def asks_for_budget(fields: list[str]) -> bool:
    return any(any(h in f.lower() for h in BUDGET_FIELD_HINTS) for f in fields or [])


def score_record(rec: dict, case: dict, gt: dict, catalog) -> dict:
    out = rec.get("outcome") or {"kind": "error", "error": "no outcome"}
    kind = out.get("kind")
    s = {"case_id": case["case_id"], "config": rec["config"], "rep": rec["rep"], "category": case["category"],
         "expected": gt["expected"], "kind": kind, "success": False, "reason": None,
         "plan_delivered": kind == "plan", "over_budget": None, "violates": None, "unverifiable": None,
         "ungrounded": None, "plan_total": None}
    if kind == "plan":
        items = out.get("items") or []
        j = judge_plan(items, case, catalog)
        evidence = {e for i in items for e in i.get("evidence_ids", [])}
        missing_evidence = evidence - catalog.evidence_ids(evidence)
        qty_ok = sum(int(i.get("quantity", 1)) for i in items) == int(case.get("quantity", 1))
        s.update(over_budget=j["over_budget"], violates=j["violates"], unverifiable=j["unverifiable"],
                 ungrounded=bool(j["unknown_offer_ids"] or missing_evidence), plan_total=j["total"],
                 plan_items=j["items"], missing_evidence=sorted(missing_evidence))
        if gt["expected"] == "clarify":
            s["reason"] = "plan_without_required_input"
        elif gt["expected"] == "no_feasible_plan":
            s["reason"] = "plan_when_infeasible"
        elif j["unknown_offer_ids"]:
            s["reason"] = "unknown_offer_id"
        elif j["wrong_category"]:
            s["reason"] = "wrong_category"
        elif not qty_ok:                       # before cost: a wrong quantity is wrong at any price
            s["reason"] = "wrong_quantity"
        elif j["violates"]:
            s["reason"] = "violates_requirement"
        elif j["over_budget"]:
            s["reason"] = "over_budget"
        elif j["unverifiable"]:
            s["reason"] = "unverifiable_claim"
        elif missing_evidence:
            s["reason"] = "unknown_evidence_id"
        else:
            s["success"], s["reason"] = True, "ok"
    elif kind == "ask_user":
        if gt["expected"] == "plan":
            s["reason"] = "unnecessary_clarification"
        elif gt["expected"] == "clarify" and not asks_for_budget(out.get("required_fields")):
            s["reason"] = "clarification_misses_budget"
        else:
            s["success"], s["reason"] = True, "ok"
    else:
        s["reason"] = "error:" + str(rec.get("stop_reason") or out.get("error") or "unknown")[:80]
    return s


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
        for exp in ("plan", "no_feasible_plan", "clarify"):
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
            "bvr": {"n": sum(1 for s in known_total if s["over_budget"]), "m": len(plans),
                    "rate": (round(sum(1 for s in known_total if s["over_budget"]) / len(plans), 3) if plans else None)},
            "plans_delivered": len(plans),
            "constraint_violation": rate(plans, "violates"), "unverifiable_claim": rate(plans, "unverifiable"),
            "ungrounded_ids": rate(plans, "ungrounded"),
            "amount_error_rate": None,
            "outcome_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "wall_s": {"p50": _pct(walls, 0.5), "p95": _pct(walls, 0.95), "max": _pct(walls, 1.0)},
            "model_calls_mean": _mean(r.get("model_calls") for r in recs),
            "prompt_tokens_mean": _mean(r.get("prompt_tokens") for r in recs),
            "completion_tokens_mean": _mean(r.get("completion_tokens") for r in recs),
            "invalid_actions": sum(r.get("invalid_actions", 0) for r in recs),
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
