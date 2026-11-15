"""构建含可选解释节点的单智能体状态图。"""

import json

from langgraph.graph import END, START, StateGraph

from .nodes import Nodes, RunScope
from .state import AgentState
from .progress import decision_control


def trace_value(value, limit=512):
    """把诊断值压成 ID、代码和摘要；检查点里不保存完整的工具页或长文本。"""
    if not isinstance(value, (dict, list)):
        return value
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= limit:
        return value
    if isinstance(value, list):
        return {"truncated": True, "chars": len(encoded),
                "items": [trace_value(item, limit // 2) for item in value[:5]]}
    summary = {"truncated": True, "chars": len(encoded)}
    for key in ("type", "tool_name", "status", "code", "reason", "total_count", "next_cursor",
                "candidate_set_id", "operation_id", "cached"):
        if key in value and (value[key] is None or isinstance(value[key], (str, int, float, bool))):
            summary[key] = value[key]
    for key in ("offer_id", "variant_id", "answer"):
        if isinstance(value.get(key), str):
            summary[key] = value[key][:128]
    return summary


def build_graph(scope: RunScope, checkpointer):
    """注册节点及条件边，并绑定用于用户问答恢复的检查点存储。"""
    nodes = Nodes(scope)
    graph = StateGraph(AgentState)
    for name in ("prepare", "guard", "decide", "execute_tool", "validate_plan",
                 "prepare_question", "suspend", "apply_answer", "explain", "finalize"):
        handler = getattr(nodes, name)

        # 默认参数固定本轮处理函数，避免循环闭包全部引用最后一个节点。
        async def execute(state, handler=handler):
            update = await handler(state)
            details = update.pop("_trace", {})
            # 历史消息本身已在 history 中；诊断不再重复保存整条 action/proposal/message。
            entry = {"node": handler.__name__, "control": decision_control(state, scope.policy),
                     "usage": update.get("usage", state["usage"]), "next_node": update.get("next_node"),
                     **{key: trace_value(value) for key, value in details.items()}}
            if handler.__name__ == "validate_plan":
                entry["proposal"] = trace_value(state["action"])
            for key in ("action", "issues", "stop_reason", "status"):
                if key in update:
                    entry[key] = trace_value(update[key])
            # 节点完成后保存累计执行时间和事件序号；等待事件的预留序号不能被覆盖。
            return {**update, "trace": (state.get("trace", []) + [entry])[-scope.policy.max_trace_entries:],
                    "active_elapsed_s": scope.elapsed,
                    "event_sequence": max(scope.sequence, update.get("event_sequence", 0))}

        graph.add_node(name, execute)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "guard")
    graph.add_conditional_edges("guard", lambda state: state["next_node"], ["decide", "prepare_question", "finalize"])
    graph.add_conditional_edges("decide", lambda state: state["next_node"],
                                ["guard", "execute_tool", "validate_plan", "prepare_question", "finalize"])
    graph.add_conditional_edges("execute_tool", lambda state: state["next_node"], ["guard", "finalize"])
    graph.add_conditional_edges("validate_plan", lambda state: state["next_node"], ["guard", "explain", "finalize"])
    graph.add_edge("explain", "finalize")
    # 问题先写入检查点再中断；恢复后进入回答校验，而不是直接结束整个任务。
    graph.add_conditional_edges("prepare_question", lambda state: state["next_node"], ["suspend", "finalize"])
    graph.add_edge("suspend", "apply_answer")
    graph.add_conditional_edges("apply_answer", lambda state: state["next_node"],
                                ["guard", "prepare_question", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
