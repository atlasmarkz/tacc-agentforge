"""构建含可选解释节点的单智能体状态图。"""

from langgraph.graph import END, START, StateGraph

from .nodes import Nodes, RunScope
from .state import AgentState


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
            # 节点完成后保存累计执行时间和事件序号；等待事件的预留序号不能被覆盖。
            return {**update, "active_elapsed_s": scope.elapsed,
                    "event_sequence": max(scope.sequence, update.get("event_sequence", 0))}

        graph.add_node(name, execute)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "guard")
    graph.add_conditional_edges("guard", lambda state: state["next_node"], ["decide", "finalize"])
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
