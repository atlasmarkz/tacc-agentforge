"""Bounded shopping-agent runtime with explicitly injected capabilities."""

from .contracts import (
    AgentDependencies, AgentEvent, AgentRequest, AgentResult, CallContext,
    CapabilityError, Issue, ModelPort, ModelReply, ModelRequest, PendingOperation,
    PlanValidationRequest, PlanValidator, ToolExecutor, ValidationResult, VerifiedPlan,
)
from .policies import RunPolicy
from .runtime import AgentRuntime
from .tools import ToolRegistry, ToolSpec

__all__ = [
    "AgentDependencies", "AgentEvent", "AgentRequest", "AgentResult",
    "AgentRuntime", "CallContext", "CapabilityError", "Issue", "ModelPort",
    "ModelReply", "ModelRequest", "PendingOperation", "PlanValidationRequest",
    "PlanValidator", "RunPolicy", "ToolExecutor", "ToolRegistry", "ToolSpec",
    "ValidationResult", "VerifiedPlan",
]
