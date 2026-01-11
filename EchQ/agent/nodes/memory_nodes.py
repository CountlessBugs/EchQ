from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

from EchQ.utils.datetime_utils import DatetimeUtils

# 防止循环引用
if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_state import AgentState
    from ..agent_memory import AgentMemory


def retrieve_memories_node(self: Agent, state: AgentState) -> AgentState:
    """检索记忆节点

    从记忆中检索与消息列表中最后一轮对话的相关信息, 并将其作为系统消息添加到消息列表中
    """
    retrieved_docs = self._memory.retrieve_similar_memories(
        query=state.values["messages"][-1].content,
        k=2,
        score_threshold=0.8
    )

    formatted_memories = [
        f"[{DatetimeUtils.format_relative_time(m.metadata['timestamp'])}] {m.page_content}" 
        for m in retrieved_docs
    ]

    if formatted_memories:
        memory_message_content = "<memory>" + "\n---\n".join(formatted_memories) + "</memory>"
        memory_message = SystemMessage(content=memory_message_content)
        return {"messages": [memory_message]}
    
    return {}


__all__ = ["retrieve_memories_node"]
