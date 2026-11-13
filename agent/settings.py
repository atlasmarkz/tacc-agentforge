"""宿主侧运行配置；库级 RunPolicy 默认值不隐式读取文件。"""

from dataclasses import dataclass
import os
from typing import Literal

from pydantic import Field, model_validator

from .config import AgentConfig, load_config
from .contracts import Schema
from .policies import RunPolicy


class ModelSettings(Schema):
    name: str | None
    base_url: str | None
    address_file: str | None
    model_path: str | None
    temperature: float = Field(ge=0)
    enable_thinking: bool | None
    constrain_output: bool
    timeout_margin_s: float = Field(ge=0)
    max_attempts: int = Field(strict=True, ge=1)
    backoff_base_s: float = Field(gt=0)
    backoff_max_s: float = Field(gt=0)
    connect_timeout_s: float = Field(gt=0)
    context_window_tokens: int | None = Field(default=None, strict=True, gt=0)

    def port_options(self):
        return self.model_dump(include={"temperature", "enable_thinking", "constrain_output", "timeout_margin_s"})

    def client_options(self):
        return self.model_dump(include={"max_attempts", "backoff_base_s", "backoff_max_s", "connect_timeout_s"})


class CheckpointSettings(Schema):
    backend: Literal["memory", "sqlite"]
    path: str
    sqlite_storage_verified: bool
    cleanup_on_start: bool
    terminal_retention_s: float = Field(ge=0)
    expired_question_grace_s: float = Field(ge=0)

    def retention(self):
        from .checkpoint_cleanup import RetentionPolicy
        return RetentionPolicy(self.terminal_retention_s, self.expired_question_grace_s)


class RuntimeValues(Schema):
    agent: RunPolicy
    model: ModelSettings
    checkpoint: CheckpointSettings
    local: dict

    @model_validator(mode="after")
    def check_window(self):
        window = self.model.context_window_tokens
        if window is not None and max(self.agent.max_output_tokens, self.agent.max_explain_tokens) >= window:
            raise ValueError("Output token budget must leave room for input in the model context window")
        return self


@dataclass(frozen=True)
class RuntimeSettings:
    config: AgentConfig
    values: RuntimeValues

    @property
    def policy(self):
        return self.values.agent

    @property
    def model(self):
        return self.values.model

    @property
    def checkpoint(self):
        return self.values.checkpoint


def load_runtime_settings(path=None, profile=None, *, default_profile=None):
    config = load_config(path or os.environ.get("AGENT_CONFIG"), profile or os.environ.get("AGENT_PROFILE"),
                         default_profile=default_profile)
    return RuntimeSettings(config, RuntimeValues.model_validate(config.values))
