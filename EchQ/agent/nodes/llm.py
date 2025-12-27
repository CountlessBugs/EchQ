from __future__ import annotations
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage


# 只有在类型检查时才导入，运行时不导入，防止循环引用
if TYPE_CHECKING:
    from ..agent import Agent, AgentState

async def call_llm_node(self: Agent, state: AgentState) -> AgentState:
    """调用 LLM 节点"""
    if self._llm is None:
        raise ValueError('LLM 未初始化，请先调用 initialize 方法')

    # 如果有待处理的消息，则添加到状态中
    if self._pending_messages:
        state['messages'].extend(self._pending_messages)
        self._pending_messages.clear()

    # 调用 LLM 生成响应
    response = await self._llm.ainvoke(state['messages'])
    usage = getattr(response, 'usage_metadata', {})
    token_usage = usage.get('total_tokens', 0)
    
    return {'messages': [response], 'token_usage': token_usage}

async def summarize_context_node(self: Agent, state: AgentState) -> AgentState:
    """总结上下文节点"""
    if self._llm is None:
        raise ValueError('LLM 未初始化，请先调用 initialize 方法')

    # 构建总结请求的消息
    history_text = '\n'.join(
        f"{m.type}: {m.content}" for m in state['messages'] if m.type in ['human', 'ai']
    )
    messages = [
        SystemMessage(content=(
            'You are a summary assistant. '
            'Your ONLY task is to produce a concise summary of the following conversation '
            'in its original language. '
            'Do NOT extend the dialogue, answer questions, or generate new sentences. '
            'Output the summary and NOTHING else. '
            'Use the nickname from the message prefix in place of "user", and replace "assistant" with "you".'
        )),
        HumanMessage(content=f'Conversation: """\n{history_text}"""')
    ]
    
    # 使用较低的温度获取总结
    response = await self._llm.ainvoke(messages, temperature=0.3)
    summary = response.content
    usage = getattr(response, 'usage_metadata', {})
    token_usage = usage.get('completion_tokens', 0)

    # 清除现有上下文，并将摘要作为系统消息添加回上下文中
    remove_messages = [RemoveMessage(id=msg.id) for msg in state['messages']]
    new_messages = [
        SystemMessage(content=self.llm_prompt),
        SystemMessage(content=f'<context_summary>\n{summary}\n</context_summary>')
    ]

    return {'messages': remove_messages + new_messages, 'token_usage': token_usage}

__all__ = ['call_llm_node', 'summarize_context_node']
