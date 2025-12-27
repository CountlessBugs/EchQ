import types

from langgraph.graph import StateGraph, START, END

from ..agent import agent, Agent, AgentState
from ..nodes.llm import call_llm_node, summarize_context_node, summarize_context_branch


# 统一定义哪些函数需要挂载, 以及挂载到 agent 上的名字
NODES = {
    'call_llm': call_llm_node,
    'summarize_context': summarize_context_node,
}

BRANCHES = {
    'summarize_context': summarize_context_branch,
}

builder = StateGraph(AgentState)

# 将节点函数挂载到 agent 实例上并添加到 workflow 中
for node_name, func in NODES.items():
    attr_name = '_' + node_name + '_node'
    setattr(agent, attr_name, types.MethodType(func, agent))

    func.__globals__['Agent'] = Agent
    func.__globals__['AgentState'] = AgentState

    builder.add_node(node_name, getattr(agent, attr_name))

# 将分支函数挂载到 agent 实例上
for branch_name, func in BRANCHES.items():
    attr_name = '_' + branch_name + '_branch'
    setattr(agent, attr_name, types.MethodType(func, agent))
    
    func.__globals__['Agent'] = Agent
    func.__globals__['AgentState'] = AgentState


# 添加桥接节点
builder.add_node('has_pending_messages_branch_to_summarize_context_branch', lambda state: state)

# 添加边
builder.add_edge(START, 'call_llm')
builder.add_conditional_edges('call_llm', agent._has_pending_messages_branch, {
    True: 'call_llm',
    False: 'has_pending_messages_branch_to_summarize_context_branch'
})
builder.add_conditional_edges('has_pending_messages_branch_to_summarize_context_branch', agent._summarize_context_branch, {
    True: 'summarize_context',
    False: END
})
builder.add_edge('summarize_context', END)

workflow = builder.compile()


__all__ = ['workflow']
