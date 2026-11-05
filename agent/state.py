"""Graph state contains JSON values; dependencies and cancellation live outside it."""

from typing import TypedDict

from .contracts import AgentRequest, Usage


class AgentState(TypedDict):
    state_version: str
    request: dict
    history: list[dict]
    usage: dict
    consecutive_action_errors: int
    rejected_plans: int
    action: dict | None
    verified_plan: dict | None
    clarification: dict | None
    issues: list[dict]
    status: str | None
    stop_reason: str | None
    next_node: str
    result: dict | None


def initial_state(request: AgentRequest) -> AgentState:
    return {
        "state_version": "1", "request": request.model_dump(mode="json"),
        "history": [], "usage": Usage().model_dump(mode="json"),
        "consecutive_action_errors": 0, "rejected_plans": 0,
        "action": None, "verified_plan": None, "clarification": None,
        "issues": [], "status": None, "stop_reason": None,
        "next_node": "guard", "result": None,
    }
