"""Build SFT demonstrations from development cases only (todo Task 7).

Inputs
- Cases: a JSON list of {"case_id", "split", "category", "message", "constraints"?}. Only split "dev" is
  accepted, and only categories the snapshot defines (the unseen fifth category stays out until the agent,
  prompts and tools are frozen). Anything else is refused, never silently dropped.
- Traces: JSONL, one recorded harness run per line:
  {"case_id", "run_id", "status", "stop_reason", "grounded": bool | null,
   "calls": [{"messages": [...chat messages exactly as sent to the model...], "reply": "<action JSON>",
              "accepted": bool}]}
  `messages` must be what the model saw in deployment (inference.model_port.to_chat_messages), recorded
  without test-only hints, so training and serving see the same prompt.

Output
- One chat example per accepted model call of a kept run: the recorded messages followed by the assistant
  turn (the action, re-serialised as compact JSON). Kept runs: a plan whose IDs were grounded in the
  snapshot, or a clarification question. Calls the harness rejected are never targets.

Checks before anything is written: no held-out case ID or held-out message text appears in any example,
exact duplicates are removed, and every target parses as JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

KEEP_OUTCOMES = {("incomplete", "validation_unavailable"), ("completed", None), ("waiting_user", None)}


class DataError(ValueError):
    """The inputs would produce training data that must not be used."""


@dataclass
class SFTExample:
    """One chat-format demonstration: messages end with the target assistant turn."""
    case_id: str
    messages: list[dict]
    metadata: dict = field(default_factory=dict)


def seen_categories(category_schemas: Path) -> set[str]:
    data = json.loads(Path(category_schemas).read_text(encoding="utf-8"))
    return set(data.get("categories", data))


def load_dev_cases(path: Path, allowed_categories: set[str] | None = None) -> list[dict]:
    """Read development cases; MUST refuse held-out cases and the unseen category."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise DataError("case file must be a JSON list")
    ids = set()
    for case in cases:
        for key in ("case_id", "split", "category", "message"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise DataError(f"case {case.get('case_id')!r}: missing {key}")
        if case["split"] != "dev":
            raise DataError(f"case {case['case_id']}: split {case['split']!r} is not allowed in training data")
        if allowed_categories is not None and case["category"] not in allowed_categories:
            raise DataError(f"case {case['case_id']}: category {case['category']!r} is not a seen category")
        if case["case_id"] in ids:
            raise DataError(f"duplicate case_id {case['case_id']}")
        ids.add(case["case_id"])
    return cases


def split_case_file(path: Path, allowed_categories: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """evaluations/cases.jsonl (all splits): dev cases for training, all other cases for the leakage check.
    Dev cases go through the same checks as load_dev_cases."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [r for r in rows if r.get("split") == "dev"]
    tmp = Path(path).with_suffix(".dev-only.tmp.json")
    try:
        tmp.write_text(json.dumps(dev, ensure_ascii=False), encoding="utf-8")
        dev = load_dev_cases(tmp, allowed_categories)
    finally:
        tmp.unlink(missing_ok=True)
    return dev, [r for r in rows if r.get("split") != "dev"]


def load_traces(path: Path) -> list[dict]:
    traces = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                traces.append(json.loads(line))
            except ValueError as exc:
                raise DataError(f"trace line {n} is not JSON") from exc
    return traces


def _keep_run(trace: dict) -> bool:
    if "score_success" in trace:
        # Verdict of the independent evaluation scorer (evaluations/score.py): the run reached the correct
        # outcome for its case. Runs where the harness rejected any step are not demonstrations either.
        return trace["score_success"] is True and not trace.get("rejected")
    if (trace.get("status"), trace.get("stop_reason")) not in KEEP_OUTCOMES:
        return False
    if trace.get("status") in ("completed", "incomplete"):
        return trace.get("grounded") is True          # a proposed plan must use real snapshot IDs
    return True


def build_examples(cases: list[dict], traces: list[dict]) -> list[SFTExample]:
    """Turn recorded runs of dev cases into tool-use demonstrations (accepted actions of good runs only)."""
    by_id = {c["case_id"]: c for c in cases}
    examples, seen = [], set()
    for trace in traces:
        case = by_id.get(trace.get("case_id"))
        if case is None:
            raise DataError(f"trace {trace.get('run_id')!r} belongs to case {trace.get('case_id')!r}, "
                            "which is not a loaded dev case")
        if not _keep_run(trace):
            continue
        for step, call in enumerate(trace.get("calls", [])):
            if not call.get("accepted"):
                continue
            try:
                action = json.loads(call["reply"])
            except (KeyError, ValueError) as exc:
                raise DataError(f"run {trace.get('run_id')} step {step}: accepted reply is not JSON") from exc
            target = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            messages = [dict(m) for m in call["messages"]] + [{"role": "assistant", "content": target}]
            key = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            examples.append(SFTExample(case["case_id"], messages, {
                "run_id": trace.get("run_id"), "step": step, "action_type": action.get("type"),
                "category": case["category"], "outcome": trace.get("status"), "sha256": key}))
    return examples


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_no_leakage(examples: list[SFTExample], heldout_cases: list[dict]) -> None:
    """Refuse if a held-out case ID or a held-out request text appears anywhere in the examples."""
    ids = {c["case_id"] for c in heldout_cases}
    texts = [_normalise(c["message"]) for c in heldout_cases if len(c.get("message", "")) >= 12]
    for ex in examples:
        if ex.case_id in ids:
            raise DataError(f"example from held-out case {ex.case_id}")
        blob = _normalise(" ".join(m.get("content", "") for m in ex.messages))
        for text in texts:
            if text in blob:
                raise DataError(f"held-out request text found in an example of case {ex.case_id}")


def write_jsonl(examples: list[SFTExample], out_path: Path) -> dict:
    """Write examples (deterministic order) and a manifest next to them; returns the manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(examples, key=lambda e: (e.case_id, e.metadata.get("run_id") or "", e.metadata.get("step", 0)))
    lines = [json.dumps({"case_id": e.case_id, "messages": e.messages, "metadata": e.metadata},
                        ensure_ascii=False, sort_keys=True) for e in ordered]
    body = ("\n".join(lines) + "\n") if lines else ""
    out_path.write_text(body, encoding="utf-8")
    actions = {}
    for e in ordered:
        actions[e.metadata.get("action_type")] = actions.get(e.metadata.get("action_type"), 0) + 1
    manifest = {"file": out_path.name, "examples": len(ordered), "cases": len({e.case_id for e in ordered}),
                "action_types": actions, "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, help="evaluations/cases.jsonl: dev cases train, the rest check leakage")
    ap.add_argument("--dev-cases", type=Path)
    ap.add_argument("--heldout-cases", type=Path,
                    help="held-out cases (any split label); only used to check for leakage")
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--category-schemas", type=Path, default=Path("configs/category_schemas.json"))
    ap.add_argument("--out", type=Path, default=Path("data/sft/train.jsonl"))
    a = ap.parse_args()
    if a.cases:
        cases, heldout = split_case_file(a.cases, seen_categories(a.category_schemas))
    elif a.dev_cases and a.heldout_cases:
        cases = load_dev_cases(a.dev_cases, seen_categories(a.category_schemas))
        heldout = json.loads(a.heldout_cases.read_text(encoding="utf-8"))
    else:
        ap.error("give --cases, or both --dev-cases and --heldout-cases")
    examples = build_examples(cases, load_traces(a.traces))
    check_no_leakage(examples, heldout)
    print(json.dumps(write_jsonl(examples, a.out), indent=2))


if __name__ == "__main__":
    main()
