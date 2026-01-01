from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage, RemoveMessage

# 只有在类型检查时才导入，运行时不导入，防止循环引用
if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_state import AgentState


def cleanup_node(self: Agent, state: AgentState) -> AgentState:
    """子图执行后的清理节点"""
    # 移除待移除消息
    current_message_ids = {m.id for m in state.get('messages', []) if m.id}
    message_ids_to_remove = state.get('message_ids_to_remove', [])
    messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in message_ids_to_remove if msg_id in current_message_ids]

    return { 'messages': messages_to_remove }


def has_tool_calls_branch(self: Agent, state: AgentState) -> bool:
    """检查智能体是否有待处理的工具调用, 并打印日志

    Returns:
        如果有待处理的工具调用则返回 True, 否则返回 False
    """
    if not state['messages'] or not isinstance(state['messages'][-1], AIMessage):
        return False
    
    last_msg = state['messages'][-1]
    if last_msg.tool_calls:
        print(f'调用工具: {last_msg.tool_calls}')
    return len(last_msg.tool_calls) > 0


__all__ = ['cleanup_node', 'has_tool_calls_branch']
