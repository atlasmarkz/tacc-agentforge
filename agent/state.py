"""定义可写入检查点的图状态；客户端、依赖对象和取消信号由运行作用域持有。"""

from typing import TypedDict

from .contracts import AgentRequest, Usage


class AgentState(TypedDict):
    """保存单次任务的请求、历史、路由和恢复信息；各节点返回局部字段更新。"""

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
    # 恢复使用原策略，防止新 Runtime 的默认配置重置执行上限。
    policy: dict
    pending_question: dict | None
    resume_answer: dict | None
    question_count: int
    answer_errors: int
    # 只累计实际执行时间；用户等待通过 pending_question 中的过期时间单独限制。
    active_elapsed_s: float
    event_sequence: int


def initial_state(request: AgentRequest, policy) -> AgentState:
    """仅为新任务初始化状态；恢复已有任务时必须读取检查点，不能重新归零。"""
    return {
        "state_version": "2", "request": request.model_dump(mode="json"),
        "history": [], "usage": Usage().model_dump(mode="json"),
        "consecutive_action_errors": 0, "rejected_plans": 0,
        "action": None, "verified_plan": None, "clarification": None,
        "issues": [], "status": None, "stop_reason": None,
        "next_node": "guard", "result": None,
        "policy": policy.model_dump(mode="json"), "pending_question": None,
        "resume_answer": None, "question_count": 0, "answer_errors": 0,
        "active_elapsed_s": 0.0, "event_sequence": 0,
    }
