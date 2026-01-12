from __future__ import annotations
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from utils.datetime_utils import DatetimeUtils

# 防止循环引用
if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_state import AgentState
    from ..agent_memory import AgentMemory


# === 记忆数据模型 ===

class MemoryItem(BaseModel):
    content: str = Field(description="需要记忆的内容, 一般为一两句话")
    type: str = Field(description="内容的类型标签(英文), 例如 preference, fact, event 等")
    importance: float = Field(description="重要程度评分，范围 0.0 到 1.0。重要性越高则越容易回忆起且遗忘越慢，重要性为 1.0 则永不遗忘", gt=0.0, le=1.0)
    # TODO: 计划加入情感标签
    # emotion: Literal["none", "happy", "sad", "angry", ...] = Field(description="与内容相关的情感标签")

class Memories(BaseModel):
    items: list[MemoryItem] = Field(description="需要记忆的内容列表，如果没有需要记忆的内容则返回空列表")


# === 记忆节点 ===

async def memorize_node(self: Agent, state: AgentState) -> AgentState:
    """记忆存储节点

    调用 LLM 提取当前对话的最后一条消息中的重要内容并标注, 存储到记忆中
    建议在总结节点之后使用此节点
    """
    if self._llm is None:
        raise ValueError("LLM 未初始化，请先调用 initialize 方法")
    
    # 构建提取请求的消息
    last_message = state["messages"][-1].content
    messages = [
        SystemMessage(content=(
            "你是一个记忆提取助手，"
            "你的任务是从用户与智能体的对话中提取出值得记忆的信息，以存储到长期记忆中。"
            "请提取出对话中有价值或有趣的内容，并将其转述为简短的一句或一段话。"
            "你需要以智能体的视角进行转述，以“我”代指智能体。"
            "你可以选取多条信息加入记忆，也可以不选取任何信息。"
        )),
        HumanMessage(content=f"<conversation>\n{last_message}\n</conversation>")
    ]
    
    # 格式化输出, 便于获取类型和重要性评分
    structured_llm = self._llm.with_structured_output(Memories)

    # 使用特定温度获取记忆内容
    response = await structured_llm.with_config(tags=["memorize"]).ainvoke(messages, temperature=1.0)

    # 存储记忆内容
    for item in response.items:
        self._memory.store_memory(
            content=item.content,
            type=item.type,
            importance=item.importance
        )
    
    return {}

def recall_node(self: Agent, state: AgentState) -> AgentState:
    """回忆节点

    从记忆中检索与消息列表中最后一轮对话的相关信息, 并将其作为系统消息添加到消息列表中
    """
    retrieved_docs = self._memory.retrieve_similar_memories(
        query=state["messages"][-1].content,
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


__all__ = ["memorize_node", "recall_node"]
