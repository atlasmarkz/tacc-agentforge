"""Replace the prompts and processing below with your own project logic."""
import json
import os
from pathlib import Path
import time
from qwen_client import chat


def main():
    base_url = os.environ["QWEN_BASE_URL"]
    prompts = ["37加58等于多少？只输出数字。", "中国的首都是哪里？只输出城市名称。"]
    records = []
    for prompt in prompts:
        started = time.perf_counter()
        result = chat([{"role": "user", "content": prompt}], base_url,
                      max_tokens=64, temperature=0)
        elapsed = round(time.perf_counter() - started, 3)
        answer = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print("Question:", prompt)
        print("Answer:", answer)
        print("Usage:", usage)
        print("Elapsed seconds:", elapsed)
        records.append({"prompt": prompt, "answer": answer,
                        "usage": usage, "elapsed_seconds": elapsed})

    relative = Path("CSIT6000V-project-results") / f"job-{os.environ['SLURM_JOB_ID']}.json"
    destination = Path(os.environ["TACC_USERDIR"]) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("RESULT_FILE=" + relative.as_posix())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}")
