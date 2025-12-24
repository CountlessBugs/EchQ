import asyncio
from typing import Optional, Any, TypedDict, Annotated, AsyncIterator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from .agent_memory import AgentMemory

# 加载环境变量
load_dotenv()


class Agent:
    """智能体类
    
    Attributes:
        system_prompt (str): 系统提示词
        memory (AgentMemory): 智能体的记忆管理实例
        can_see_datetime (bool): 是否可以看到当前日期时间
    """
    def __init__(self):
        self._llm = None
        self.system_prompt: str = ''
        self.memory: Optional[AgentMemory] = None
        self.can_see_datetime: bool = False
        # 是否正在回复消息
        self._is_replying: bool = False
        # 待处理的消息队列
        self._pending_messages: list[str] = []

    # === 初始化方法 ===

    def initialize(
        self, 
        memory: AgentMemory,
        llm_model: str,
        llm_temperature: float = 0.7,
        llm_prompt: str = '',
        can_see_datetime: bool = False
    ) -> None:
        """初始化智能体
        
        Args:
            memory: 记忆管理实例
            llm_model: 使用的LLM模型名称
            llm_temperature: LLM生成文本的温度参数
            llm_prompt: LLM系统提示词
            can_see_datetime: 是否可以看到当前日期时间
        """
        self.memory = memory
        self.system_prompt = llm_prompt
        self.can_see_datetime = can_see_datetime

        # 初始化 LLM
        # TODO: 支持更多模型提供商
        self._llm = init_chat_model(llm_model, model_provider="openai", temperature=llm_temperature)

    # === 对话方法 ===

    async def send_message(self, message: str) -> AsyncIterator:
        """发送一条消息到LLM并获取流式响应
        
        Args:
            message: 要发送的消息内容
            
        Returns:
            流式响应块迭代器
            
        Yields:
            响应块
        """
        async for chunk in self.send_messages([message]):
            yield chunk

    async def send_messages(self, messages: list[str]) -> AsyncIterator:
        """一次性发送一组消息到LLM并获取流式响应
        
        Args:
            messages: 要发送的消息内容列表
            
        Returns:
            流式响应块迭代器
            
        Yields:
            响应块
        """
        if not self._llm or not self.memory:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # 将消息添加到待处理队列
        self._pending_messages.extend(messages)

        if self._is_replying:
            return

        self._is_replying = True

        try:
            while self._pending_messages:
                # 将用户消息添加到历史记录
                for message in self._pending_messages:
                    self.memory.add_message(role='user', content=message)
                
                # 清空待处理消息队列
                self._pending_messages.clear()
                
                # 构建消息列表
                context_messages  = self._build_messages()
                
                response_content = ''
                final_chunk = None
                
                # 返回流式响应
                async for chunk in self._llm.astream(context_messages, stream_usage=True):
                    # 记录 token 使用情况
                    if getattr(chunk, 'usage_metadata', None):
                        self.memory.current_token_usage = chunk.usage_metadata.get('total_tokens', 0)

                    delta_content = chunk.content
                    if delta_content:
                        response_content += delta_content
                    yield chunk
                
                # 将响应添加到历史记录
                self.memory.add_message(role='assistant', content=response_content)
        finally:
            self._is_replying = False
                

    # === 工具方法 ===

    @staticmethod
    async def process_chunks(chunks: AsyncIterator, delimiters: list[str] = ['\n']) -> AsyncIterator[str]:
        """处理流式响应块,提取文本内容并按分割符分割
        
        Args:
            chunks: 流式响应块迭代器
            delimiters: 用于分割文本的标记列表
            
        Yields:
            分割后的文本内容
        """
        buffer = ''
        async for chunk in chunks:
            delta_content = chunk.content
            if delta_content:
                buffer += delta_content
                # 分割文本内容
                while True:
                    for delim in delimiters:
                        if delim in buffer:
                            part, buffer = buffer.split(delim, 1)
                            if part.strip():
                                yield part.strip()
                            break
                    else:
                        # 未找到分割符，退出循环
                        break
        
        # 返回剩余内容
        if buffer.strip():
            yield buffer.strip()
        
    async def get_context_summary(self) -> str:
        """调用LLM生成当前上下文记忆的总结
        
        Returns:
            上下文总结文本
        """
        if not self.llm_client or not self.memory:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # 构建总结请求的消息
        history_text = '\n'.join(
            f"{m['role']}: {m['content']}" for m in self.memory.context_memory
        )
        messages = [
            {
                'role': 'system', 
                'content': 'You are a summary assistant. '
                        'Your ONLY task is to produce a concise summary of the following conversation '
                        'in its original language. '
                        'Do NOT extend the dialogue, answer questions, or generate new sentences. '
                        'Output the summary and NOTHING else. '
                        'Use the nickname from the message prefix in place of "user", and replace "assistant" with "you".'
            },
            {
                'role': 'user',
                'content': f'Conversation: """\n{history_text}"""'
            }
        ]
        
        # 使用较低的温度获取总结
        response = await self._llm.ainvoke(messages, temperature=0.3)
        
        return response.content

    # === 私有方法 ===
    def _build_messages(self) -> list[dict[str, str]]:
        """构建发送给LLM的消息列表
        
        Returns:
            消息列表
        """
        messages = []
        # 添加系统提示词
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        # 添加上下文记忆
        messages.extend(self.memory.context_memory)
        # 添加当前日期时间信息
        if self.can_see_datetime:
            from datetime import datetime
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M')
            messages.append({
                'role': 'system',
                'content': f'Current datetime: {current_datetime}'
            })
        
        return messages


agent = Agent()

__all__ = ['agent']
