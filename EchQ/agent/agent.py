from openai import OpenAI
from typing import Iterator, Optional, Dict, Any, List

from .agent_memory import AgentMemory


class LLMClient:
    """LLM客户端类, 负责与LLM API的交互
    
    Attributes:
        api_url (str): LLM API 基础 URL
        api_key (str): LLM API 密钥
        model (str): 使用的LLM模型名称
        temperature (float): LLM生成文本的温度参数
    """
    
    def __init__(self,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float = 1.3
    ) -> None:
        """初始化LLM客户端
        
        Args:
            api_url: API基础URL
            api_key: API密钥
            model: 模型名称
            temperature: 温度参数
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=api_url)
    
    def chat_completion_stream(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None
    ) -> Iterator:
        """发送聊天请求并返回流式响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数 (可选, 覆盖默认值)
            
        Returns:
            流式响应迭代器
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            stream=True
        )
        return response
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """发送聊天请求并返回完整响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数 (可选, 覆盖默认值)
            
        Returns:
            完整响应对象
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            stream=False
        )
        return response


class Agent:
    """智能体类
    
    Attributes:
        llm_client (LLMClient): LLM客户端实例
        system_prompt (str): 系统提示词
        memory (AgentMemory): 智能体的记忆管理实例
        can_see_datetime (bool): 是否可以看到当前日期时间
    """
    def __init__(self):
        self.llm_client: Optional[LLMClient] = None
        self.system_prompt: str = ''
        self.memory: Optional[AgentMemory] = None
        self.can_see_datetime: bool = False

    def initialize(
        self, 
        memory: AgentMemory, 
        llm_api_url: str, 
        llm_api_key: str, 
        llm_model: str, 
        llm_temperature: float = 0.7, 
        llm_prompt: str = '',
        can_see_datetime: bool = False
    ) -> None:
        """初始化智能体
        
        Args:
            memory: 记忆管理实例
            llm_api_url: LLM API 基础URL
            llm_api_key: LLM API 密钥
            llm_model: 使用的LLM模型名称
            llm_temperature: LLM生成文本的温度参数
            llm_prompt: LLM系统提示词
            can_see_datetime: 是否可以看到当前日期时间
        """
        self.memory = memory
        self.system_prompt = llm_prompt
        self.can_see_datetime = can_see_datetime
        self.llm_client = LLMClient(
            api_url=llm_api_url,
            api_key=llm_api_key,
            model=llm_model,
            temperature=llm_temperature
        )

    def _build_messages(self) -> List[Dict[str, str]]:
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

    def send_message(self, message: str) -> Iterator:
        """发送消息到LLM并获取流式响应
        
        Args:
            message: 要发送的消息内容
            
        Returns:
            流式响应块迭代器
            
        Yields:
            响应块
        """
        if not self.llm_client or not self.memory:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # 将用户消息添加到历史记录
        self.memory.add_message(role='user', content=message)
        
        # 构建消息列表
        messages = self._build_messages()
        
        # 发送消息并获取流式响应
        response_stream = self.llm_client.chat_completion_stream(messages)
        
        response_content = ''
        final_chunk = None
        
        # 返回流式响应
        for chunk in response_stream:
            final_chunk = chunk
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                response_content += delta_content
            yield chunk
        
        # 将响应添加到历史记录
        self.memory.add_message(role='assistant', content=response_content)
        
        # 记录 token 使用情况
        if final_chunk and hasattr(final_chunk, 'usage') and final_chunk.usage:
            self.memory.current_token_usage = final_chunk.usage.total_tokens

    @staticmethod
    def process_chunks(chunks: Iterator, delimiters: List[str] = ['\n']) -> Iterator[str]:
        """处理流式响应块,提取文本内容并按分割符分割
        
        Args:
            chunks: 流式响应块迭代器
            delimiters: 用于分割文本的标记列表
            
        Yields:
            分割后的文本内容
        """
        buffer = ''
        for chunk in chunks:
            delta_content = chunk.choices[0].delta.content
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
        
    def get_context_summary(self) -> str:
        """获取当前上下文记忆的总结
        
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
        response = self.llm_client.chat_completion(messages, temperature=0.3)
        
        return response.choices[0].message.content


agent = Agent()

__all__ = ['agent']
