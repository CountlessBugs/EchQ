import types

from langgraph.graph import StateGraph, START, END

from ..agent import agent, Agent
from ..agent_state import AgentState
from ..nodes.basic_nodes import invoke_type_branch, has_tool_calls_branch, has_pending_messages_branch
from ..nodes.llm_nodes import call_llm_node, summarize_context_node, summarize_context_branch


# 统一定义哪些函数需要挂载, 以及挂载到 agent 上的名字
NODES = {
    "call_llm": call_llm_node,
    "summarize_context": summarize_context_node,
}

BRANCHES = {
    "invoke_type": invoke_type_branch,
    "has_tool_calls": has_tool_calls_branch,
    "has_pending_messages": has_pending_messages_branch,
    "summarize_context": summarize_context_branch,
}

builder = StateGraph(AgentState)

# 将节点函数挂载到 agent 实例上并添加到 workflow 中
for node_name, func in NODES.items():
    attr_name = "_" + node_name + "_node"
    setattr(agent, attr_name, types.MethodType(func, agent))

    func.__globals__["Agent"] = Agent
    func.__globals__["AgentState"] = AgentState

    builder.add_node(node_name, getattr(agent, attr_name))

# 将分支函数挂载到 agent 实例上
for branch_name, func in BRANCHES.items():
    attr_name = "_" + branch_name + "_branch"
    setattr(agent, attr_name, types.MethodType(func, agent))
    
    func.__globals__["Agent"] = Agent
    func.__globals__["AgentState"] = AgentState


# 添加工具调用节点
if agent._tool_node is not None:
    builder.add_node("execute_tool_calls", agent._tool_node)
else:
    # 如果没有工具则添加占位节点
    builder.add_node("execute_tool_calls", lambda state: state)

# 添加桥接节点
builder.add_node("goto_has_pending_messages_branch", lambda state: state)
builder.add_node("goto_summarize_context_branch", lambda state: state)

# 添加边
builder.add_conditional_edges(START, agent._invoke_type_branch, {
    "scheduled": "call_llm",
    "user_message": "call_llm",
    "none": END
})
builder.add_conditional_edges("call_llm", agent._has_tool_calls_branch, {
    True: "execute_tool_calls",
    False: "goto_has_pending_messages_branch"
})
builder.add_conditional_edges("goto_has_pending_messages_branch", agent._has_pending_messages_branch, {
    True: "call_llm",
    False: "goto_summarize_context_branch"
})
builder.add_conditional_edges("execute_tool_calls", agent._has_pending_messages_branch, {
    True: "call_llm",
    False: "goto_summarize_context_branch"
})
builder.add_conditional_edges("goto_summarize_context_branch", agent._summarize_context_branch, {
    True: "summarize_context",
    False: END
})
builder.add_edge("summarize_context", END)

workflow = builder.compile()


__all__ = ["workflow"]
