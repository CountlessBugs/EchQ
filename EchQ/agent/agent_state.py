from __future__ import annotations
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent 状态

        用于在 StateGraph 中存储智能体的状态、
    
    Attributes:
        messages: 对话消息列表
        token_usage: 上一次对话的 token 使用量
    """
    messages: Annotated[list[BaseMessage], add_messages]
    # 由于子图无法直接移除父图的消息，故需要层层向上传递一个待移除的消息 ID 列表
    # 每一层都需要按照这个列表移除对应的消息，最终在根图中清空该列表
    message_ids_to_remove: Annotated[list[str], add_str]
    token_usage: int


# === 自定义 Reducer 函数 ===

def add_str(left: list[str], right: list[str]) -> list[str]:
    # 如果右侧是清空标志，则返回空列表
    if right == ['__CLEAR__']:
        return []
    
    return (left or []) + (right or [])


__all__ = ['AgentState']
