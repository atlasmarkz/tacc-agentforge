"""Tests for the SFT pilot code that do not need a GPU (training/sft.py, training/lora_int4.py)."""
import json

import pytest

from training.lora_int4 import (
    LORA_TARGETS, adapter_config, module_from_peft_key, peft_key,
)
from training.sft import encode_example, load_config, split_examples


# ---- config --------------------------------------------------------------------------------------------

def write(tmp_path, text):
    p = tmp_path / "sft.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_config_expands_environment_and_parses_types(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_PATH", "/models/qwen")
    cfg = load_config(write(tmp_path, "base_model: ${QWEN_MODEL_PATH}\nbase_revision: course-v1\n"
                                      "train_file: data/sft/train.jsonl\noutput_dir: out  # comment\n"
                                      "learning_rate: 1.0e-4\nepochs: 2\nvalidation_cases: [dev-02, dev-09]\n"))
    assert cfg.base_model == "/models/qwen" and cfg.epochs == 2 and cfg.learning_rate == 1e-4
    assert cfg.validation_cases == ["dev-02", "dev-09"] and str(cfg.output_dir) == "out"
    assert load_config(write(tmp_path, "base_model: m\nbase_revision: r\ntrain_file: t\noutput_dir: o\n"),
                       {"epochs": 3}).epochs == 3


@pytest.mark.parametrize("text,match", [
    ("base_model: TODO\nbase_revision: TODO\ntrain_file: t\noutput_dir: o\n", "not set"),
    ("base_model: ${NOT_SET_ANYWHERE_X}\nbase_revision: r\ntrain_file: t\noutput_dir: o\n", "not set"),
    ("base_model: m\nbase_revision: r\ntrain_file: t\noutput_dir: o\nlearnin_rate: 1\n", "unknown config keys"),
])
def test_config_refuses_unset_or_misspelled(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_config(write(tmp_path, text))


def test_repository_config_is_complete(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_PATH", "/models/qwen")
    monkeypatch.setenv("TACC_USERDIR", "/userdir")
    cfg = load_config("configs/sft.yaml")
    assert cfg.lora_r <= 8               # the served vLLM build was tested with --max-lora-rank 8
    assert cfg.max_train_minutes <= 35   # leaves time to load, save and reload within one hour


# ---- encoding ------------------------------------------------------------------------------------------

class FakeTokenizer:
    """Mimics the chat template contract that encode_example relies on (one char = one token)."""

    def __init__(self, prefix_ok=True):
        self.prefix_ok = prefix_ok

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        assert enable_thinking is False and tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        if add_generation_prompt:
            return text + "<assistant>" + ("" if self.prefix_ok else "<think></think>")
        if messages[-1]["role"] == "assistant" and self.prefix_ok:
            return text
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
        {"role": "assistant", "content": '{"type":"ask_user"}'}]


def test_loss_covers_only_the_target_and_its_end():
    (ids, labels), why = encode_example(FakeTokenizer(), MSGS, 10_000)
    assert why is None and len(ids) == len(labels)
    target = "".join(chr(t) for t, y in zip(ids, labels) if y != -100)
    assert target == '{"type":"ask_user"}</>'
    assert all(y == -100 for y in labels[:len("<system>S</><user>U</><assistant>")])


def test_encoding_refuses_misaligned_template_long_or_targetless_examples():
    assert encode_example(FakeTokenizer(prefix_ok=False), MSGS, 10_000) == (None, "template_prefix_mismatch")
    assert encode_example(FakeTokenizer(), MSGS, 10) == (None, "too_long")
    assert encode_example(FakeTokenizer(), MSGS[:2], 10_000) == (None, "no_assistant_target")


def test_validation_split_is_by_case():
    rows = [{"case_id": c} for c in ("dev-01", "dev-02", "dev-01", "dev-09")]
    train, val = split_examples(rows, ["dev-02", "dev-09"])
    assert [r["case_id"] for r in train] == ["dev-01", "dev-01"] and len(val) == 2


# ---- adapter layout ------------------------------------------------------------------------------------

def test_peft_keys_have_the_prefix_vllm_loads():
    key = peft_key("model.language_model.layers.3.mlp.gate_proj", "A")
    assert key == "base_model.model.model.language_model.layers.3.mlp.gate_proj.lora_A.weight"
    assert module_from_peft_key(key) == ("model.language_model.layers.3.mlp.gate_proj", "A")
    # one "model." short loaded nothing in vLLM (jobs 28347-28352): refuse it on both sides
    with pytest.raises(ValueError):
        peft_key("language_model.layers.3.mlp.gate_proj", "A")
    with pytest.raises(ValueError):
        module_from_peft_key("base_model.model.language_model.layers.3.mlp.gate_proj.lora_A.weight")


def test_adapter_config_matches_the_served_format():
    cfg = adapter_config(8, 16, ["q_proj", "gate_proj", "q_proj"], "/models/qwen")
    assert cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM"
    assert cfg["target_modules"] == ["gate_proj", "q_proj"] and cfg["r"] == 8 and cfg["lora_alpha"] == 16
    assert set(LORA_TARGETS) >= {"q_proj", "in_proj_qkv", "out_proj", "down_proj"}


def test_export_and_import_round_trip_on_a_tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from training.lora_int4 import add_lora, export_peft, import_peft, lora_modules

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.gate_proj = torch.nn.Linear(6, 4, bias=False)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([Block(), Block()])

    m = Tiny()
    assert len(add_lora(m, 2, 4)) == 2
    for mod in lora_modules(m).values():
        torch.nn.init.normal_(mod.lora_B)
    x = torch.randn(3, 6)
    before = m.model.language_model.layers[1].mlp.gate_proj(x)
    info = export_peft(m, tmp_path / "a", 2, 4, "tiny")
    assert info["modules"] == 2 and json.loads((tmp_path / "a/adapter_config.json").read_text())["r"] == 2
    for mod in lora_modules(m).values():
        torch.nn.init.zeros_(mod.lora_B)
    assert import_peft(m, tmp_path / "a")["missing"] == 0
    assert torch.allclose(before, m.model.language_model.layers[1].mlp.gate_proj(x))
