import asyncio
from typing import Optional, Any, TypedDict, Annotated, AsyncIterator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

# 加载环境变量
load_dotenv()


class AgentState(TypedDict):
    """Agent 状态

        用于在 StateGraph 中存储智能体的状态、
    
    Attributes:
    """
    messages: Annotated[list, add_messages]

class Agent:
    """智能体类
    
    Attributes:
        llm_prompt: LLM 系统提示词
    """
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None
        self._llm: Optional[BaseChatModel] = None
        
        self.llm_prompt: str = ''

        self._is_replying = False
        self._pending_messages: list[BaseMessage] = []

    # === 初始化方法 ===

    def initialize(
        self,
        llm_model: str,
        llm_temperature: float = 0.7,
        llm_prompt: str = ''
    ) -> None:
        """初始化智能体
        
        Args:
            llm_model: 使用的LLM模型名称
            llm_temperature: LLM生成文本的温度参数
            llm_prompt: LLM系统提示词
        """
        self.llm_prompt = llm_prompt

        # 初始化 LLM
        # TODO: 支持更多模型提供商
        self._llm = init_chat_model(llm_model, model_provider="openai", temperature=llm_temperature)
        
        # 构建图
        self._graph = self._build_graph()

    # === 对话方法 ===
    async def send_message(self, message: str) -> AsyncIterator:
        """发送一条消息到LLM并获取响应
        
        Args:
            message: 发送的消息
            
        Yields:
            LLM 生成的响应消息片段
        """
        if self._graph is None:
            raise ValueError("智能体未初始化，请先调用 initialize 方法。")
        if self._llm is None:
            raise ValueError("LLM 未初始化，请先调用 initialize 方法。")

        if self._is_replying:
            # 如果正在回复，则将消息加入待处理队列
            self._pending_messages.append(HumanMessage(content=message))
            return
        
        # 构造配置，线程 ID 默认为0
        config = {"configurable": {"thread_id": 0}}
        
        # 获取当前状态，判断是否需要发送 SystemMessage
        state = await self._graph.aget_state(config)
        
        # 如果是新对话，则加入系统提示词
        if not state.values.get("messages"):
            input_data = {"messages": [SystemMessage(content=self.llm_prompt), HumanMessage(content=message)]}
        else:
            input_data = {"messages": [HumanMessage(content=message)]}
        
        # 执行图
        async for event in self._graph.astream_events(
            input_data,
            config=config,
            version="v2"
        ):
            # 过滤出LLM的token流事件
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if getattr(chunk, "content", None):
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

    def _build_graph(self) -> CompiledStateGraph:
        """构建智能体的状态图
        
        Returns:
            编译后的状态图
        """
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node('call_llm', self._call_llm_node)

        # 添加边
        workflow.add_edge(START, 'call_llm')
        workflow.add_conditional_edges('call_llm', self._has_pending_messages_branch, {
            True: 'call_llm',
            False: END
        })

        return workflow.compile(checkpointer=MemorySaver())
    
    # === 节点方法 ===

    async def _call_llm_node(self, state: AgentState) -> AgentState:
        """调用 LLM 节点处理消息
        
        Args:
            state: 当前智能体状态
            
        Returns:
            LLM 生成的响应消息
        """
        if self._llm is None:
            raise ValueError("LLM 未初始化，请先调用 initialize 方法。")

        self._is_replying = True

        # 如果有待处理的消息，则添加到状态中
        if self._pending_messages:
            state['messages'].extend(self._pending_messages)
            self._pending_messages.clear()

        # 调用 LLM 生成响应
        response = await self._llm.ainvoke(state['messages'])
        
        # 执行完毕，重置状态
        self._is_replying = False
        
        return {"messages": [response]}

    def _has_pending_messages_branch(self, state: AgentState) -> bool:
        """检查智能体是否有待处理的消息
        
        Args:
            state: 当前智能体状态

        Returns:
            如果有待处理的消息则返回 True, 否则返回 False
        """
        return len(self._pending_messages) > 0


agent = Agent()

__all__ = ['agent']
