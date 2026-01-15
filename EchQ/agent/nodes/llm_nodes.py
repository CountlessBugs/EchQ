from __future__ import annotations
from typing import TYPE_CHECKING, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 防止循环引用
if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_state import AgentState


async def call_llm_node(self: Agent, state: AgentState) -> AgentState:
    """调用 LLM 节点
    
        调用绑定了工具的 LLM 来生成响应消息
        该节点之后需添加工具节点来处理可能的工具调用

    """
    if self._llm is None:
        raise ValueError("LLM 未初始化，请先调用 initialize 方法")

    new_messages = self._pending_messages.copy()
    self._pending_messages.clear()

    # 如果有待处理的消息，则添加到上下文中
    if new_messages:
        messages_for_llm = list(state["messages"]) + list(new_messages)
    else:
        messages_for_llm = state["messages"]

    # 在消息末尾添加当前时间(本地时间)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")  # 格式化为字符串
    time_message = SystemMessage(content=f"<current_time>{formatted_time}</current_time>")

    # 调用 LLM 生成响应
    response = await self._llm_with_tools.with_config(tags=["chat_response"]).ainvoke(messages_for_llm + [time_message])

    # 构建新增消息列表
    new_messages.append(response)

    update_dict: dict[str, Any] = {
        "messages": new_messages
    }

    # 获取 token 使用量
    usage = getattr(response, "usage_metadata", {})
    if usage:
        update_dict["token_usage"] = usage.get("total_tokens", 0)
    
    return update_dict

async def summarize_context_node(self: Agent, state: AgentState) -> AgentState:
    """总结上下文节点"""
    if self._llm is None:
        raise ValueError("LLM 未初始化，请先调用 initialize 方法")

    # 清除已回忆记忆ID集合, 以便在新上下文中重新回忆
    self._memory.clear_recalled_memory_ids()

    # 构建总结请求的消息
    history_text = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if m.type in ["human", "ai"]
    )
    messages = [
        SystemMessage(content=(
            "You are a summary assistant. "
            "Your ONLY task is to produce a concise summary of the following conversation "
            "in its original language. "
            "Do NOT extend the dialogue, answer questions, or generate new sentences. "
            "Output the summary and NOTHING else. "
            "Use the nickname from the message prefix in place of \"user\", and replace \"assistant\" with \"you\"."
        )),
        HumanMessage(content=f"<conversation>\n{history_text}\n</conversation>")
    ]
    
    # 使用较低的温度获取总结
    response = await self._llm.with_config(tags=["summary"]).ainvoke(messages, temperature=0.3)
    summary = response.content
    usage = getattr(response, "usage_metadata", {}) or {}
    token_usage = usage.get("completion_tokens", 0)

    # 构建待移除的消息 ID 列表
    message_ids_to_remove = [m.id for m in state["messages"]]

    # 获取当前时间(本地时间)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")  # 格式化为字符串

    # 将摘要作为系统消息添加回上下文中
    new_messages = [
        SystemMessage(content=self.llm_prompt, id="system_prompt"),
        SystemMessage(content=f"<context_summary summary_time={formatted_time}>\n{summary}\n</context_summary>")
    ]

    return {
        "messages": [RemoveMessage(REMOVE_ALL_MESSAGES)] + new_messages,
        "message_ids_to_remove": message_ids_to_remove,
        "token_usage": token_usage
    }

def summarize_context_branch(self: Agent, state: AgentState) -> bool:
    """判断是否需要总结上下文的分支函数"""
    return state.get("token_usage", 0) > self.token_limit


__all__ = ["call_llm_node", "summarize_context_node", "summarize_context_branch"]
