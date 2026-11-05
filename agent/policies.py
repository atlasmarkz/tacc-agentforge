"""Validated, per-run limits. Business limits are independent of graph recursion."""

from pydantic import Field

from .contracts import Schema


class RunPolicy(Schema):
    max_model_calls: int = Field(default=8, strict=True, ge=1, le=1000)
    max_tool_calls: int = Field(default=8, strict=True, ge=0, le=1000)
    max_action_repairs: int = Field(default=2, strict=True, ge=0, le=20)
    max_plan_repairs: int = Field(default=2, strict=True, ge=0, le=20)
    deadline_seconds: float = Field(default=120, gt=0, le=3600)
    call_timeout_s: float = Field(default=30, gt=0, le=3600)
    max_context_chars: int = Field(default=16000, strict=True, ge=1000, le=1000000)
    max_history_messages: int = Field(default=12, strict=True, ge=0, le=100)
    max_output_tokens: int = Field(default=512, strict=True, ge=1, le=32768)

    @property
    def recursion_limit(self) -> int:
        return 5 * self.max_model_calls + 10
