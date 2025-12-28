import asyncio
from typing import Optional, Any, AsyncIterator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .agent_state import AgentState

# 加载环境变量
load_dotenv()


class Agent:
    """智能体类
    
    Attributes:
        llm_prompt: LLM 系统提示词
    """
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None
        self._config = {'configurable': {'thread_id': '0'}}
        self._llm: Optional[BaseChatModel] = None
        
        self.llm_prompt: str = ''
        self.token_limit: int = 16000

        self._is_busy = False
        self._pending_messages: list[BaseMessage] = []

    # === 属性方法 ===

    @property
    def context(self) -> list[BaseMessage]:
        """获取当前对话的上下文消息列表"""
        if self._graph is None:
            raise ValueError('智能体未初始化，请先调用 initialize 方法')
        state = self._graph.get_state(self._config)
        return state.values.get('messages', [])

    @property
    def token_usage(self) -> int:
        """获取上一次对话的 token 使用量"""
        if self._graph is None:
            raise ValueError('智能体未初始化，请先调用 initialize 方法')
        state = self._graph.get_state(self._config)
        return state.values.get('token_usage', 0)

    # === 初始化方法 ===

    def initialize(
        self,
        llm_model: str,
        llm_temperature: float = 0.7,
        llm_prompt: str = '',
        token_limit: int = 16000,
        *,
        workflow: Optional[CompiledStateGraph] = None
    ) -> None:
        """初始化智能体
        
        Args:
            llm_model: 使用的LLM模型名称
            llm_temperature: LLM生成文本的温度参数
            llm_prompt: LLM系统提示词
        """
        self.llm_prompt = llm_prompt
        self.token_limit = token_limit

        # 初始化 LLM
        # TODO: 支持更多模型提供商
        self._llm = init_chat_model(llm_model, model_provider='openai', temperature=llm_temperature)
        
        # 构建图
        self._graph = self._build_graph(workflow)

    # === 对话方法 ===

    async def send_message(self, message: str) -> AsyncIterator:
        """发送一条消息到LLM并获取响应
        
        Args:
            message: 发送的消息
            
        Yields:
            LLM 生成的响应消息片段
        """
        if self._graph is None:
            raise ValueError('智能体未初始化，请先调用 initialize 方法')
        if self._llm is None:
            raise ValueError('LLM 未初始化，请先调用 initialize 方法')

        if self._is_busy:
            # 如果正在回复，则将消息加入待处理队列
            self._pending_messages.append(HumanMessage(content=message))
            return
        
        # 获取当前状态，判断是否需要发送 SystemMessage
        state = await self._graph.aget_state(self._config)
        
        # 如果是新对话，则加入系统提示词
        if not state.values.get('messages'):
            input_data = {'messages': [SystemMessage(content=self.llm_prompt), HumanMessage(content=message)]}
        else:
            input_data = {'messages': [HumanMessage(content=message)]}
        
        # 执行图
        async for event in self._graph.astream_events(
            input_data,
            config=self._config,
            version='v2'
        ):
            # 过滤出带有 chat_response 标签的 LLM 的 token 流事件
            if (
                event['event'] == 'on_chat_model_stream'
                and 'chat_response' in event.get('tags', [])
            ):
                chunk = event['data']['chunk']
                if getattr(chunk, 'content', None):
                    yield chunk

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

    # === 图构建方法 ===

    def _build_graph(self, workflow: Optional[CompiledStateGraph] = None) -> CompiledStateGraph:
        """构建智能体的状态图
        
        Returns:
            编译后的状态图
        """
        builder = StateGraph(AgentState)

        if workflow is None:
            # 加载默认工作流
            from .workflows.default import workflow as default_workflow
            workflow = default_workflow

        builder.add_node('workflow', workflow)

        # 设置入口
        builder.add_conditional_edges(START, self._entry_branch, {
            True: 'workflow',
            False: END
        })
        # 设置出口
        builder.add_node('exit', self._exit_node)
        builder.add_edge('workflow', 'exit')
        builder.add_edge('exit', END)

        return builder.compile(checkpointer=MemorySaver())
    
    # === 节点方法 ===

    # 入口
    def _entry_branch(self, state: AgentState) -> bool:
        """入口分支，检查智能体是否处于可用状态，并设置忙碌标志
        
        Returns:
            如果智能体可用则返回 True, 否则返回 False
        """
        if self._llm is None:
            raise ValueError('LLM 未初始化，请先调用 initialize 方法')
        
        if self._is_busy:
            return False
        self._is_busy = True

        return True
    
    # 出口
    def _exit_node(self, state: AgentState) -> AgentState:
        """出口节点，重置忙碌标志，完成消息替换"""
        self._is_busy = False

        # 移除待移除消息
        current_message_ids = {m.id for m in state.get('messages', []) if m.id}
        message_ids_to_remove = state.get('message_ids_to_remove', [])
        messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in message_ids_to_remove if msg_id in current_message_ids]

        return { 'messages': messages_to_remove, 'message_ids_to_remove': ['__CLEAR__'] }

    def _has_pending_messages_branch(self, state: AgentState) -> bool:
        """检查智能体是否有待处理的消息

        Returns:
            如果有待处理的消息则返回 True, 否则返回 False
        """
        return len(self._pending_messages) > 0


agent = Agent()

__all__ = ['agent', 'Agent', 'AgentState']
