"""SFT pilot: train, save, and reload a LoRA adapter (todo Task 7).

Input: chat JSONL from training/prepare_data.py (messages end with the target assistant action).
Each example is rendered with the model's own chat template, thinking off, exactly as the serving adapter
sends it; the loss covers only the assistant action and its end token. Examples longer than max_seq_len are
dropped (a target is never cut). A step that runs out of GPU memory is skipped and counted. Training stops at
max_train_minutes so that saving and the reload check still fit in a one-hour TACC job.

Output (adapter_dir): adapter_model.safetensors + adapter_config.json in the peft layout vLLM serves on this
model, and training_meta.json (base model, hashes of config/tokenizer/data, per-step losses, timing).
The reload check compares the logits of the trained model with the logits after re-loading the saved file.

This is a pilot: it shows the pipeline works. It makes no quality claim; comparing SFT with its own base
model needs the frozen held-out evaluation (Task 6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class SFTConfig:
    base_model: str            # exact checkpoint; record revision for provenance
    base_revision: str
    train_file: Path
    output_dir: Path
    lora_r: int = 8
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    epochs: int = 1
    max_seq_len: int = 8192
    seed: int = 0
    max_train_minutes: float = 30.0
    validation_cases: list = field(default_factory=list)    # case_ids held out for validation loss
    grad_clip: float = 1.0


def _parse_scalar(text: str):
    t = text.strip()
    if t.startswith(("'", '"')) and t.endswith(t[0]):
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        return [_parse_scalar(x) for x in t[1:-1].split(",") if x.strip()]
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return {"true": True, "false": False, "null": None}.get(t.lower(), t)


def load_config(path: Path, overrides: dict | None = None) -> SFTConfig:
    """Flat key: value YAML (configs/sft.yaml). ${VAR} is expanded from the environment."""
    raw = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"cannot parse config line: {line!r}")
        raw[key.strip()] = _parse_scalar(os.path.expandvars(value))
    raw.update(overrides or {})
    known = {f.name for f in fields(SFTConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "TODO" in (str(raw.get("base_model")), str(raw.get("base_revision"))) or "$" in str(raw.get("base_model", "")):
        raise ValueError("base_model/base_revision are not set (or an environment variable is missing)")
    raw["train_file"], raw["output_dir"] = Path(raw["train_file"]), Path(raw["output_dir"])
    return SFTConfig(**raw)


def encode_example(tok, messages: list[dict], max_len: int):
    """(input_ids, labels) with the prompt masked, or (None, reason). The full rendering must start with the
    prompt rendering, otherwise the mask would not line up with the target."""
    if not messages or messages[-1].get("role") != "assistant":
        return None, "no_assistant_target"
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        return None, "template_prefix_mismatch"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if not target_ids:
        return None, "empty_target"
    ids = prompt_ids + target_ids
    if len(ids) > max_len:
        return None, "too_long"
    return (ids, [-100] * len(prompt_ids) + target_ids), None


def split_examples(rows: list[dict], validation_cases) -> tuple[list[dict], list[dict]]:
    val = set(validation_cases or [])
    return [r for r in rows if r["case_id"] not in val], [r for r in rows if r["case_id"] in val]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(cfg: SFTConfig) -> dict:
    base = Path(cfg.base_model)
    tok_files = sorted(p for p in base.glob("tokenizer*") if p.is_file()) + sorted(base.glob("*chat_template*"))
    return {"base_model": str(base), "base_revision": cfg.base_revision,
            "config_sha256": sha256_file(base / "config.json") if (base / "config.json").exists() else None,
            "tokenizer_sha256": {p.name: sha256_file(p) for p in tok_files},
            "train_file": str(cfg.train_file), "train_sha256": sha256_file(cfg.train_file)}


def _loss(model, ids, labels, device):
    import torch
    x = torch.tensor([ids], device=device)
    y = torch.tensor([labels], device=device)
    return model(input_ids=x, labels=y).loss


def train(cfg: SFTConfig) -> Path:
    """Run SFT and return the adapter directory (with base model + tokenizer info)."""
    import torch
    from transformers import AutoTokenizer
    from .lora_int4 import add_lora, export_peft, import_peft, load_model, lora_modules

    t_start = time.monotonic()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    adapter_dir = Path(cfg.output_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(adapter_dir / "train_log.jsonl", "w", encoding="utf-8")

    def log(event, **data):
        rec = {"t": round(time.monotonic() - t_start, 1), "event": event, **data}
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        print(json.dumps(rec), flush=True)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    rows = [json.loads(l) for l in Path(cfg.train_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_rows, val_rows = split_examples(rows, cfg.validation_cases)
    dropped = {}

    def encode_all(rs):
        out = []
        for r in rs:
            enc, why = encode_example(tok, r["messages"], cfg.max_seq_len)
            if enc is None:
                dropped[why] = dropped.get(why, 0) + 1
            else:
                out.append({"case_id": r["case_id"], "ids": enc[0], "labels": enc[1]})
        return out

    train_set, val_set = encode_all(train_rows), encode_all(val_rows)
    lengths = sorted(len(e["ids"]) for e in train_set)
    log("data", rows=len(rows), train=len(train_set), validation=len(val_set), dropped=dropped,
        tokens_median=lengths[len(lengths) // 2] if lengths else None, tokens_max=lengths[-1] if lengths else None,
        target_tokens=sum(sum(1 for y in e["labels"] if y != -100) for e in train_set))
    if not train_set:
        raise SystemExit("no usable training examples")

    model, load_info = load_model(cfg.base_model)
    wrapped = add_lora(model, cfg.lora_r, cfg.lora_alpha)
    device = next(model.parameters()).device
    log("model", **load_info, wrapped=len(wrapped),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)

    def validation_loss():
        if not val_set:
            return None
        model.eval()
        with torch.no_grad():
            vals = [_loss(model, e["ids"], e["labels"], device).item() for e in val_set]
        model.train()
        return round(sum(vals) / len(vals), 4)

    log("validation", epoch=0, loss=validation_loss())
    model.train()
    steps, oom, stopped = 0, 0, None
    budget = cfg.max_train_minutes * 60
    t_train = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_set)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for i in order:
            if time.monotonic() - t_train > budget:
                stopped = "time_budget"
                break
            e = train_set[i]
            t0 = time.monotonic()
            try:
                loss = _loss(model, e["ids"], e["labels"], device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optim.step()
            except torch.cuda.OutOfMemoryError:
                oom += 1
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log("oom", tokens=len(e["ids"]))
                continue
            optim.zero_grad(set_to_none=True)
            steps += 1
            log("step", epoch=epoch, step=steps, loss=round(loss.item(), 4), tokens=len(e["ids"]),
                seconds=round(time.monotonic() - t0, 2),
                peak_gib=[round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())])
        log("validation", epoch=epoch, loss=validation_loss())
        if stopped:
            break

    meta = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "provenance": _provenance(cfg),
            "steps": steps, "oom_skipped": oom, "stopped": stopped or "completed",
            "train_minutes": round((time.monotonic() - t_train) / 60, 1), "dropped": dropped,
            "examples": {"train": len(train_set), "validation": len(val_set)}}
    exported = export_peft(model, adapter_dir, cfg.lora_r, cfg.lora_alpha, cfg.base_model, meta)
    log("saved", **exported, path=str(adapter_dir))
    meta["reload"] = reload_adapter(adapter_dir, model, (val_set or train_set)[:2], device, log)
    (adapter_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_file.close()
    return adapter_dir


def reload_adapter(adapter_dir: Path, model=None, samples=(), device=None, log=print):
    """Smoke test: load the saved adapter on its base model and check it reproduces the trained logits.
    Within the training job the base is already loaded: zero the LoRA weights, load the saved file, and
    compare logits on a few samples with those of the trained model."""
    import torch
    from .lora_int4 import import_peft, lora_modules
    if model is None:
        raise ValueError("reload in a fresh process needs the base model; run within train() or serve with vLLM")
    model.eval()

    def logits():
        with torch.no_grad():
            return [model(input_ids=torch.tensor([s["ids"]], device=device)).logits.float().cpu() for s in samples]

    before = logits()
    with torch.no_grad():
        for mod in lora_modules(model).values():
            mod.lora_A.zero_()
            mod.lora_B.zero_()
    zeroed = logits()
    info = import_peft(model, adapter_dir)
    after = logits()
    diff = max((a - b).abs().max().item() for a, b in zip(before, after)) if samples else None
    changed = max((a - z).abs().max().item() for a, z in zip(before, zeroed)) if samples else None
    result = {**info, "max_abs_logit_diff_after_reload": diff, "max_abs_logit_change_without_adapter": changed,
              "ok": info["missing"] == 0 and diff is not None and diff < 1e-2}
    log("reload", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--set", action="append", default=[], help="override, e.g. --set epochs=2")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        k, _, v = item.partition("=")
        overrides[k.strip()] = _parse_scalar(v)
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
