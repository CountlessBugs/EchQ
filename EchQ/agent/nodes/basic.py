from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage

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


__all__ = ['cleanup_node']
