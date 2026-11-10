"""Recommended harness RunPolicy settings for this deployment (Qwen3.8-27B-AWQ-INT4 on vLLM, 16K window).

Measured with the real knowledge tools on TACC (issue #6; jobs 28560 and 28561):
- With the harness defaults, 512 output tokens are used up by reasoning when thinking is on (empty replies,
  finish_reason "length"), and 16000 context characters drop earlier steps so the model repeats searches.
- With 36000 characters (about 12K tokens, inside the 16384-token model window) and 2048 output tokens when
  thinking, neither failure occurred.

Returned as plain keyword arguments so this module does not import the harness:
    RunPolicy(**run_policy_kwargs(enable_thinking=False))
Whether these become the harness's own defaults is for the harness owner to decide (issue #6).
"""
from __future__ import annotations

MODEL_WINDOW_TOKENS = 16384
CONTEXT_CHARS = 36000
OUTPUT_TOKENS_THINKING_OFF = 512      # decisions measured at about 30 to 200 tokens
OUTPUT_TOKENS_THINKING_ON = 2048      # reasoning tokens count against the output limit


def run_policy_kwargs(enable_thinking: bool | None) -> dict:
    """`enable_thinking=None` means the model default, which for Qwen3 is thinking on."""
    thinking = enable_thinking is not False
    return {
        "max_context_chars": CONTEXT_CHARS,
        "max_output_tokens": OUTPUT_TOKENS_THINKING_ON if thinking else OUTPUT_TOKENS_THINKING_OFF,
    }
