"""购物智能体的公开接口；模型、工具和校验能力均由调用方显式注入。"""

from .contracts import (
    AgentDependencies, AgentEvent, AgentRequest, AgentResult, CallContext,
    CapabilityError, Issue, ModelPort, ModelReply, ModelRequest, PendingOperation,
    PendingQuestion, PlanValidationRequest, PlanValidator, ResumeRequest, RunStateError,
    ToolExecutor, ValidationResult, VerifiedPlan,
)
from .policies import RunPolicy
from .runtime import AgentRuntime
from .tools import ToolRegistry, ToolSpec

__all__ = [
    "AgentDependencies", "AgentEvent", "AgentRequest", "AgentResult",
    "AgentRuntime", "CallContext", "CapabilityError", "Issue", "ModelPort",
    "ModelReply", "ModelRequest", "PendingOperation", "PlanValidationRequest",
    "PlanValidator", "RunPolicy", "ToolExecutor", "ToolRegistry", "ToolSpec",
    "ValidationResult", "VerifiedPlan", "PendingQuestion", "ResumeRequest", "RunStateError",
]
