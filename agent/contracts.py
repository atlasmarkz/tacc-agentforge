"""定义与具体服务商无关的边界类型，所有持久化值都可表示为 JSON。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Protocol, Union, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

if TYPE_CHECKING:
    from .tools import ToolRegistry


class Schema(BaseModel):
    """所有边界模型的基类，禁止未知字段并限制字段重新赋值。"""

    # frozen 不会递归冻结列表等容器，跨适配器传递需要隔离的数据仍应深拷贝。
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
ToolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Amount = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]
ConstraintField = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]


class AgentRequest(Schema):
    """描述一次智能体运行请求及其所属的会话和数据快照。"""

    schema_version: Literal["1"] = "1"
    run_id: Identifier
    conversation_id: Identifier
    snapshot_id: Identifier
    message: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    constraints: dict[str, JsonValue] = Field(default_factory=dict)


class PlanItem(Schema):
    """计划中的单个商品条目及其数量和证据引用。"""

    offer_id: Identifier
    quantity: int = Field(strict=True, gt=0, le=10000)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=20)


class ProposedPlan(Schema):
    """模型提出的、尚未经过业务校验的采购计划。"""

    items: list[PlanItem] = Field(min_length=1, max_length=20)


class CallTool(Schema):
    """请求运行时调用指定工具的动作。"""

    type: Literal["call_tool"]
    tool_name: ToolName
    arguments: dict[str, JsonValue]


class ProposePlan(Schema):
    """请求运行时提交采购计划的动作。"""

    type: Literal["propose_plan"]
    plan: ProposedPlan


class AskUser(Schema):
    """请求用户补充信息后再继续执行的动作。"""

    type: Literal["ask_user"]
    question: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    required_fields: list[ConstraintField] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def check_fields(self):
        """只允许询问互不重复的顶层约束字段，排除身份及运行控制字段。"""
        reserved = {"run_id", "conversation_id", "snapshot_id", "schema_version",
                    "message", "constraints", "policy", "usage"}
        if len(set(self.required_fields)) != len(self.required_fields) or reserved.intersection(self.required_fields):
            raise ValueError("ask_user must name unique, top-level constraint fields")
        return self


class PendingQuestion(Schema):
    """已写入检查点的稳定问题；恢复时必须匹配其 ID。"""

    question_id: Identifier
    question: str
    required_fields: list[ConstraintField]
    expires_at: float = Field(gt=0)  # UTC 纪元秒，用于独立限制用户等待时间。


class ResumeRequest(Schema):
    """恢复一个等待中的问题；answer 只包含被询问的约束字段。"""

    run_id: Identifier
    question_id: Identifier
    answer: dict[str, JsonValue]


Action = Annotated[Union[CallTool, ProposePlan, AskUser], Field(discriminator="type")]
ACTION_ADAPTER = TypeAdapter(Action)


class Issue(Schema):
    """可安全传递给调用方的问题描述。"""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(max_length=500)


class Message(Schema):
    """发送给模型的对话消息。"""

    role: Literal["system", "user", "assistant"]
    content: str


class ToolDescription(Schema):
    """提供给模型的工具名称、用途和参数定义。"""

    name: ToolName
    description: str
    parameters: dict[str, JsonValue]


class ModelRequest(Schema):
    """发送给模型端口的标准化请求。"""

    messages: list[Message]
    tools: list[ToolDescription]
    # None requests free text without constrained decoding; decision requests supply a schema.
    action_schema: dict[str, JsonValue] | None = None
    max_output_tokens: int = Field(gt=0)


class ModelReply(Schema):
    """模型端口返回的文本结果及用量信息。"""

    content: str = Field(max_length=65536)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    finish_reason: Literal["stop", "length", "error"] = "stop"


class PendingOperation(Schema):
    """表示异步操作尚未完成的句柄。"""

    handle: str = Field(min_length=1, max_length=256)


@dataclass(frozen=True)
class CallContext:
    """一次模型、工具或校验器调用共享的运行上下文。"""

    run_id: str
    snapshot_id: str
    operation_id: str
    timeout_s: float
    cancellation: asyncio.Event
    # Invoked on the event-loop thread; the port isolates callback exceptions.
    on_delta: Callable[[str, str], None] | None = None


class ModelPort(Protocol):
    """模型适配器必须实现的最小接口。"""

    async def generate(
        self, request: ModelRequest, context: CallContext
    ) -> ModelReply | PendingOperation: ...


class ToolExecutor(Protocol):
    """工具适配器必须实现的最小执行接口。"""

    async def execute(
        self, arguments: BaseModel, context: CallContext
    ) -> BaseModel | dict[str, JsonValue] | PendingOperation: ...


class VerifiedPlan(Schema):
    """经过业务校验并补充金额信息的采购计划。"""

    snapshot_id: Identifier
    items: list[PlanItem] = Field(min_length=1, max_length=20)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    total_cost: Amount
    remaining_funds: Amount
    cost_breakdown: dict[str, Amount] = Field(default_factory=dict)


class EvidenceFact(Schema):
    """来自选定快照的有界证据；摘录仍是不可信的来源文本。"""

    snapshot_id: Identifier
    offer_id: Identifier
    evidence_id: Identifier
    title: str = Field(max_length=500)
    merchant: str | None = Field(default=None, max_length=300)
    unit_price: Amount | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    price_age_seconds: int | None = Field(default=None, strict=True, ge=0)
    excerpt: str = Field(default="", max_length=1400)
    source_url: str | None = Field(default=None, max_length=2048)
    scope: str | None = Field(default=None, max_length=100)


class EvidenceResolver(Protocol):
    async def resolve(self, plan: VerifiedPlan, context: CallContext) -> list[EvidenceFact]: ...


class Explanation(Schema):
    """verified 只表示模型文本通过规则检查；fallback 表示采用计划模板。"""

    text: str = Field(min_length=1, max_length=65536)
    verified: bool
    fallback: bool
    cited_evidence_ids: list[Identifier] = Field(default_factory=list)


class PlanValidationRequest(Schema):
    """提交给计划校验器的请求及其指纹。"""

    request: AgentRequest
    plan: ProposedPlan
    proposal_fingerprint: str


class ValidationResult(Schema):
    """计划校验结果，成功时必须包含已验证计划。"""

    status: Literal["passed", "failed", "unavailable"]
    proposal_fingerprint: str
    verified_plan: VerifiedPlan | None = None
    issues: list[Issue] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def check_verified_plan(self):
        """确保校验状态、已验证计划和问题列表保持一致。"""

        if (self.status == "passed") != (self.verified_plan is not None):
            raise ValueError("only a passed validation must contain a verified plan")
        if self.status == "passed" and self.issues:
            raise ValueError("a passed validation cannot contain violations")
        return self


class PlanValidator(Protocol):
    """计划校验适配器必须实现的最小接口。"""

    async def validate(
        self, request: PlanValidationRequest, context: CallContext
    ) -> ValidationResult: ...


class Usage(Schema):
    """记录一次运行中的模型、工具、修复和令牌用量。"""

    model_calls: int = 0
    explain_model_calls: int = 0
    tool_calls: int = 0
    # 修复统计记录实际拒绝次数，包含最终耗尽修复额度的那次拒绝。
    action_repairs: int = 0
    plan_repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unreported_token_calls: int = 0
    elapsed_s: float = 0.0


Status = Literal["completed", "incomplete", "failed", "cancelled", "waiting_user"]


class AgentResult(Schema):
    """本次调用的结束或等待结果，包含任务身份、产物和累计消耗。"""

    schema_version: Literal["1"] = "1"
    run_id: Identifier
    conversation_id: Identifier
    snapshot_id: Identifier
    status: Status
    stop_reason: str | None = None
    verified_plan: VerifiedPlan | None = None
    clarification: AskUser | None = None
    pending_question: PendingQuestion | None = None
    explanation: Explanation | None = None
    issues: list[Issue] = Field(default_factory=list)
    usage: Usage

    @model_validator(mode="after")
    def check_outcome(self):
        """确保最终状态与验证计划或用户澄清请求相互匹配。"""

        if (self.status == "completed") != (self.verified_plan is not None):
            raise ValueError("only completed results must contain a verified plan")
        if self.explanation is not None and self.status != "completed":
            raise ValueError("only completed results may contain an explanation")
        if (self.status == "waiting_user") != (self.clarification is not None):
            raise ValueError("only waiting_user results must contain a clarification")
        if (self.status == "waiting_user") != (self.pending_question is not None):
            raise ValueError("only waiting_user results must contain a pending question")
        return self


class AgentEvent(Schema):
    """运行期间发出的有序事件，可携带完成结果或可恢复等待结果。"""

    run_id: Identifier
    sequence: int
    node: str
    type: str
    data: dict[str, JsonValue] = Field(default_factory=dict)
    result: AgentResult | None = None


class CapabilityError(Exception):
    """适配器声明的失败；公开消息必须可以安全地展示给用户。"""

    def __init__(self, code: str, public_message: str, *, recoverable: bool = False):
        self.issue = Issue(code=code, message=public_message)
        self.recoverable = recoverable
        super().__init__(public_message)


class UnsupportedCapabilityError(NotImplementedError):
    """表示当前适配器不支持请求的能力。"""

    pass


class RunStateError(ValueError):
    """恢复或运行关联错误；拒绝请求时不会推进已有执行。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AgentDependencies:
    """运行智能体所需的模型、工具、校验器和事件接收器。"""

    model: ModelPort
    tools: ToolRegistry
    validator: PlanValidator | None = None
    event_sink: Callable[[AgentEvent], None] | None = None
    evidence_resolver: EvidenceResolver | None = None
