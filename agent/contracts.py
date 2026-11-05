"""Provider-neutral boundary types. All persisted values have JSON representations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Protocol, Union, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

if TYPE_CHECKING:
    from .tools import ToolRegistry


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
ToolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Amount = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]


class AgentRequest(Schema):
    schema_version: Literal["1"] = "1"
    run_id: Identifier
    conversation_id: Identifier
    snapshot_id: Identifier
    message: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    constraints: dict[str, JsonValue] = Field(default_factory=dict)


class PlanItem(Schema):
    offer_id: Identifier
    quantity: int = Field(strict=True, gt=0, le=10000)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=20)


class ProposedPlan(Schema):
    items: list[PlanItem] = Field(min_length=1, max_length=20)


class CallTool(Schema):
    type: Literal["call_tool"]
    tool_name: ToolName
    arguments: dict[str, JsonValue]


class ProposePlan(Schema):
    type: Literal["propose_plan"]
    plan: ProposedPlan


class AskUser(Schema):
    type: Literal["ask_user"]
    question: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    required_fields: list[str] = Field(min_length=1, max_length=20)


Action = Annotated[Union[CallTool, ProposePlan, AskUser], Field(discriminator="type")]
ACTION_ADAPTER = TypeAdapter(Action)


class Issue(Schema):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(max_length=500)


class Message(Schema):
    role: Literal["system", "user", "assistant"]
    content: str


class ToolDescription(Schema):
    name: ToolName
    description: str
    parameters: dict[str, JsonValue]


class ModelRequest(Schema):
    messages: list[Message]
    tools: list[ToolDescription]
    action_schema: dict[str, JsonValue]
    max_output_tokens: int = Field(gt=0)


class ModelReply(Schema):
    content: str = Field(max_length=65536)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    finish_reason: Literal["stop", "length", "error"] = "stop"


class PendingOperation(Schema):
    handle: str = Field(min_length=1, max_length=256)


@dataclass(frozen=True)
class CallContext:
    run_id: str
    snapshot_id: str
    operation_id: str
    timeout_s: float
    cancellation: asyncio.Event


class ModelPort(Protocol):
    async def generate(
        self, request: ModelRequest, context: CallContext
    ) -> ModelReply | PendingOperation: ...


class ToolExecutor(Protocol):
    async def execute(
        self, arguments: BaseModel, context: CallContext
    ) -> BaseModel | dict[str, JsonValue] | PendingOperation: ...


class VerifiedPlan(Schema):
    snapshot_id: Identifier
    items: list[PlanItem] = Field(min_length=1, max_length=20)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    total_cost: Amount
    remaining_funds: Amount
    cost_breakdown: dict[str, Amount] = Field(default_factory=dict)


class PlanValidationRequest(Schema):
    request: AgentRequest
    plan: ProposedPlan
    proposal_fingerprint: str


class ValidationResult(Schema):
    status: Literal["passed", "failed", "unavailable"]
    proposal_fingerprint: str
    verified_plan: VerifiedPlan | None = None
    issues: list[Issue] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def check_verified_plan(self):
        if (self.status == "passed") != (self.verified_plan is not None):
            raise ValueError("only a passed validation must contain a verified plan")
        if self.status == "passed" and self.issues:
            raise ValueError("a passed validation cannot contain violations")
        return self


class PlanValidator(Protocol):
    async def validate(
        self, request: PlanValidationRequest, context: CallContext
    ) -> ValidationResult: ...


class Usage(Schema):
    model_calls: int = 0
    tool_calls: int = 0
    action_repairs: int = 0
    plan_repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unreported_token_calls: int = 0
    elapsed_s: float = 0.0


Status = Literal["completed", "incomplete", "failed", "cancelled", "waiting_user"]


class AgentResult(Schema):
    schema_version: Literal["1"] = "1"
    run_id: Identifier
    conversation_id: Identifier
    snapshot_id: Identifier
    status: Status
    stop_reason: str | None = None
    verified_plan: VerifiedPlan | None = None
    clarification: AskUser | None = None
    issues: list[Issue] = Field(default_factory=list)
    usage: Usage

    @model_validator(mode="after")
    def check_outcome(self):
        if (self.status == "completed") != (self.verified_plan is not None):
            raise ValueError("only completed results must contain a verified plan")
        if (self.status == "waiting_user") != (self.clarification is not None):
            raise ValueError("only waiting_user results must contain a clarification")
        return self


class AgentEvent(Schema):
    run_id: Identifier
    sequence: int
    node: str
    type: str
    data: dict[str, JsonValue] = Field(default_factory=dict)
    result: AgentResult | None = None


class CapabilityError(Exception):
    """An adapter-declared failure. public_message must be safe to show to the user."""

    def __init__(self, code: str, public_message: str, *, recoverable: bool = False):
        self.issue = Issue(code=code, message=public_message)
        self.recoverable = recoverable
        super().__init__(public_message)


class UnsupportedCapabilityError(NotImplementedError):
    pass


@dataclass(frozen=True)
class AgentDependencies:
    model: ModelPort
    tools: ToolRegistry
    validator: PlanValidator | None = None
    event_sink: Callable[[AgentEvent], None] | None = None
