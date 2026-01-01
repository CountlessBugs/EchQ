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
    message_ids_to_remove: Annotated[list[str], add_to_list]
    token_usage: int

    # 工具调用结果列表
    tool_call_results: Annotated[list[ToolCallResult], add_unique_dict]

class ToolCallResult(TypedDict):
    """工具调用结果字典类型

        支持的结果类型包括文本、图片、音频和文件. 文本结果直接存储内容, 图片、音频和文件结果存储对应的 URL.
        如果有多个结果, 则在工具实现中返回多个 ToolCallResult 字典.

    Attributes:
        tool_name: 工具名称
        id: 工具调用 ID
        type: 结果类型 (text | image | record | file)
        content: 结果内容
    """
    tool_name: str
    id: str
    type: str
    content: str


# === 自定义 Reducer 函数 ===

CLEAR = '__CLEAR__'  # 清空标志

def add_to_list(left: list, right: list) -> list:
    """将右侧列表添加到左侧列表中, 支持清空操作"""
    # 如果右侧是清空标志，则返回空列表
    if right == [CLEAR]:
        return []
    
    return (left or []) + (right or [])

def add_unique_dict(left: list[dict], right: list[dict]) -> list[dict]:
    """将右侧列表添加到左侧列表中, 并根据 dict 中的 id 字段去重, 支持清空操作

        dict 需要包含 id 字段, 否则不会被添加
    """
    # 如果右侧是清空标志，则返回空列表
    if right == [CLEAR]:
        return []
    
    if left is not None:
        existing_ids = {d['id'] for d in left if 'id' in d}
        right = [d for d in right if 'id' in d and d['id'] not in existing_ids]

    return (left or []) + (right or [])


__all__ = ['AgentState', 'ToolCallResult', CLEAR]
