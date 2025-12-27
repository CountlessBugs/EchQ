import types

from langgraph.graph import StateGraph, START, END

from ..agent import agent, AgentState
from ..nodes.llm import call_llm_node, summarize_context_node


# 统一定义哪些函数需要挂载，以及挂载到 agent 上的名字
NODE_MAPPING = {
    '_call_llm_node': call_llm_node,
    '_summarize_context_node': summarize_context_node
}

# 将节点函数挂载到 agent 实例上
for attr_name, func in NODE_MAPPING.items():
    setattr(agent, attr_name, types.MethodType(func, agent))


builder = StateGraph(AgentState)

# 添加节点
builder.add_node('call_llm', agent._call_llm_node)
builder.add_node('summarize_context', agent._summarize_context_node)

# 添加边
builder.add_edge(START, 'call_llm')
builder.add_conditional_edges('call_llm', agent._has_pending_messages_branch, {
    True: 'call_llm',
    False: END
})

workflow = builder.compile()

__all__ = ['workflow']
