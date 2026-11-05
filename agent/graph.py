"""Explicit, single-agent state graph. All loops pass through the guard."""

from langgraph.graph import END, START, StateGraph

from .nodes import Nodes, RunScope
from .state import AgentState


def build_graph(scope: RunScope):
    nodes = Nodes(scope)
    graph = StateGraph(AgentState)
    for name in ("prepare", "guard", "decide", "execute_tool", "validate_plan", "suspend", "finalize"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "guard")
    graph.add_conditional_edges("guard", lambda state: state["next_node"], ["decide", "finalize"])
    graph.add_conditional_edges("decide", lambda state: state["next_node"],
                                ["guard", "execute_tool", "validate_plan", "suspend", "finalize"])
    for name in ("execute_tool", "validate_plan"):
        graph.add_conditional_edges(name, lambda state: state["next_node"], ["guard", "finalize"])
    graph.add_edge("suspend", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
