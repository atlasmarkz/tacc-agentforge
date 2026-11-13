"""Build deploy/tacc/train-sft-serve/check_prompts.json: the first-step prompt of every dev case, exactly as the
deployed harness sends it (no test-only category hint), for comparing the base model with an SFT adapter.

- system message: copied from a recorded harness trace (identical in all of them: harness prompt, tools, schema)
- user message: the harness's AgentRequest JSON for the case
- expected first action: from the case itself, not from any model
    call_tool search_products with filters.category == the case category, or ask_user when the budget is missing

  python scripts/make_sft_check_prompts.py --cases cases.jsonl --traces traces.jsonl
"""
import argparse
import json
from pathlib import Path


def build(cases: list[dict], traces: list[dict], trained_cases: set[str], validation_cases=()) -> dict:
    system = {t["calls"][0]["messages"][0]["content"] for t in traces if t.get("calls")}
    if len(system) != 1:
        raise SystemExit("the traces do not agree on the system message")
    system = system.pop()
    schema_text = system[system.index("Action JSON Schema:") + len("Action JSON Schema:"):].strip()
    prompts = []
    for c in cases:
        if c["split"] != "dev":
            raise SystemExit("only dev cases may be used here; the held-out split stays sealed")
        constraints = dict(c["constraints"])
        constraints["budget_basis"] = c.get("budget_basis", "pre_tax")
        user = json.dumps({"schema_version": "1", "run_id": f"eval-{c['case_id']}-check-0",
                           "conversation_id": f"c-eval-{c['case_id']}-check-0",
                           "snapshot_id": "snapshot_5bcd00723fb0b7589cbe50c8", "message": c["message"],
                           "constraints": constraints}, ensure_ascii=False)
        expected = ({"type": "ask_user"} if c["constraints"].get("purchase_cap") is None
                    else {"type": "call_tool", "tool_name": "search_products", "category": c["category"]})
        prompts.append({"case_id": c["case_id"], "category": c["category"], "expected": expected,
                        # train: the adapter was trained on runs of this case; validation: in the file but
                        # held back from training; unseen: no successful run, so not in the file at all
                        "role": ("validation" if c["case_id"] in validation_cases else
                                 "train" if c["case_id"] in trained_cases else "unseen"),
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    return {"schema": json.loads(schema_text), "prompts": prompts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--train", type=Path, help="training JSONL: marks which dev cases the adapter was trained on")
    ap.add_argument("--validation", default="", help="comma-separated case IDs held back from training")
    ap.add_argument("--out", type=Path, default=Path("deploy/tacc/train-sft-serve/check_prompts.json"))
    a = ap.parse_args()
    cases = [c for c in (json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip())
             if c["split"] == "dev"]
    traces = [json.loads(l) for l in a.traces.read_text(encoding="utf-8").splitlines() if l.strip()]
    trained = ({json.loads(l)["case_id"] for l in a.train.read_text(encoding="utf-8").splitlines() if l.strip()}
               if a.train else set())
    out = build(cases, traces, trained, set(filter(None, a.validation.split(","))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    roles = [p["role"] for p in out["prompts"]]
    print(f"{len(roles)} prompts: {roles.count('train')} train, {roles.count('validation')} validation, "
          f"{roles.count('unseen')} unseen -> {a.out}")


if __name__ == "__main__":
    main()
