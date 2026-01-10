import asyncio
from typing import Optional, TypedDict, Literal, AsyncIterator
import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from .agent_state import AgentState, CLEAR

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class ImageMessage(TypedDict):
    text: str
    images: Optional[list[str]] # 图片 url

class Agent:
    """智能体类
    
    Attributes:
        llm_prompt: LLM 系统提示词
    """
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None
        self._config = {"configurable": {"thread_id": "0"}}
        self._llm: Optional[BaseChatModel] = None
        self._llm_with_tools: Optional[BaseChatModel] = None
        self._tools: dict[str, BaseTool] = {}
        self.tool_node: Optional[ToolNode] = None
        
        self.llm_prompt: str = ""
        self.token_limit: int = 16000
        self.vision_enabled: bool = False  # 是否启用视觉能力(上传图片)

        self._is_busy = False
        self._pending_messages: list[BaseMessage] = []

    # === 属性方法 ===

    @property
    def context(self) -> list[BaseMessage]:
        """获取当前对话的上下文消息列表"""
        if self._graph is None:
            raise ValueError("智能体未初始化，请先调用 initialize 方法")
        state = self._graph.get_state(self._config)
        return state.values.get("messages", [])

    @property
    def token_usage(self) -> int:
        """获取上一次对话的 token 使用量"""
        if self._graph is None:
            raise ValueError("智能体未初始化，请先调用 initialize 方法")
        state = self._graph.get_state(self._config)
        return state.values.get("token_usage", 0)

    # === 初始化方法 ===

    def initialize(
        self,
        llm_model: str,
        llm_temperature: float = 0.7,
        llm_prompt: str = "",
        token_limit: int = 16000,
        *,
        workflow: Optional[CompiledStateGraph] = None,
        tools: Optional[list[BaseTool]] = None,
        llm_model_provider: str = "openai",
        enable_vision: bool = False
    ) -> None:
        """初始化智能体
        
        Args:
            llm_model: 使用的LLM模型名称
            llm_temperature: LLM生成文本的温度参数
            llm_prompt: LLM系统提示词
            token_limit: 对话的最大token限制
            workflow: 自定义工作流图, 如果为 None 则使用默认工作流
            tools: 智能体可用的工具列表
            llm_model_provider: LLM模型提供商名称
            enable_vision: 是否启用视觉能力(图片输入)
        """
        self.llm_prompt = llm_prompt
        self.token_limit = token_limit
        self.vision_enabled = enable_vision

        # 初始化 LLM
        self._llm = init_chat_model(llm_model, model_provider=llm_model_provider, temperature=llm_temperature)
        
        # 绑定工具
        if tools is not None:
            self._tools = {tool.name: tool for tool in tools}
            self._llm_with_tools = self._llm.bind_tools(tools)
            self.tool_node = ToolNode(tools)
        else:
            self._llm_with_tools = self._llm

        # 构建图
        self._graph = self._build_graph(workflow)

        # 初始化状态
        initial_state: AgentState = {
            "messages": [SystemMessage(content=self.llm_prompt, id="system_prompt")],
        }
        self._graph.update_state(self._config, initial_state)

        self._is_busy = False

    # === 对话方法 ===

    async def invoke(
        self,
        invoke_type: Literal["scheduled", "user_message"],
        message: str | ImageMessage | list[str | ImageMessage] | None = None
    ) -> AsyncIterator:
        """激活 Agent 并获取响应
        
        Args:
            invoke_type: 调用类型, 可选值为 "scheduled" 或 "user_message"
            message: 发送的消息
            
        Yields:
            Agent 生成的响应消息片段
        """
        if self._graph is None:
            raise ValueError("智能体未初始化，请先调用 initialize 方法")
        if self._llm is None:
            raise ValueError("LLM 未初始化，请先调用 initialize 方法")

        # 将消息统一转换为列表进行处理
        if message is not None and not isinstance(message, list):
            message = [message]

        # 将消息加入待处理队列
        if message is not None:
            for msg in message:
                if isinstance(msg, str):
                    self._pending_messages.append(HumanMessage(content=msg))
                elif isinstance(msg, dict):
                    text = msg.get("text", "")
                    images = msg.get("images", None)
                    if images and self.vision_enabled:
                        msg_content = [
                            {"type": "text", "text": text},
                            *[{"type": "image_url", "image_url": {"url": img}} for img in images]
                        ]
                        self._pending_messages.append(HumanMessage(content=msg_content))
                    else:
                        self._pending_messages.append(HumanMessage(content=text))

        # 防止并发调用
        if self._is_busy:
            return        
        self._is_busy = True
        
        try:
            # 准备输入数据
            input_data = {
                "invoke_type": invoke_type,
                "messages": self._pending_messages.copy()
            }

            # 清空待处理消息队列
            self._pending_messages.clear()

            # 执行图
            async for event in self._graph.astream_events(
                input_data,
                config=self._config,
                version="v2"
            ):
                # 过滤出带有 chat_response 标签的 LLM 的 token 流事件
                if (
                    event["event"] == "on_chat_model_stream"
                    and "chat_response" in event.get("tags", [])
                ):
                    chunk = event["data"]["chunk"]
                    yield chunk

                # 捕获工具执行结束事件
                elif event["event"] == "on_tool_end":
                    await asyncio.sleep(0.05)  # 确保工具结果已写入状态
                    state = self._graph.get_state(self._config)
                    tool_results = state.values.get("tool_call_results", [])
                    
                    if tool_results:     
                        # 返回工具调用结果
                        for result in tool_results:
                            yield result

                        # 清空工具调用结果
                        self._graph.update_state(self._config, {"tool_call_results": [CLEAR]})

        finally:
            self._is_busy = False

    # === 工具方法 ===

    @staticmethod
    async def process_chunks(chunks: AsyncIterator, delimiters: list[str] = ["\n"]) -> AsyncIterator[str]:
        """处理流式响应块, 提取文本内容并按分割符分割, 工具调用结果保持不变
        
        Args:
            chunks: 流式响应块迭代器
            delimiters: 用于分割文本的标记列表
            
        Yields:
            分割后的文本内容
        """
        buffer = ""
        async for chunk in chunks:
            # 判断是否为工具调用结果, 如果是则直接 yield
            if isinstance(chunk, dict) and "tool_name" in chunk:
                yield chunk
                continue

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
            from .workflows.default_wf import workflow as default_workflow
            workflow = default_workflow

        builder.add_node("workflow", workflow)

        # 设置入口
        builder.add_edge(START, "workflow")
        # 设置出口
        builder.add_node("exit", self._exit_node)
        builder.add_edge("workflow", "exit")
        builder.add_edge("exit", END)

        return builder.compile(checkpointer=MemorySaver())
    
    # === 节点方法 ===
    
    # 出口
    def _exit_node(self, state: AgentState) -> AgentState:
        """出口节点，完成消息替换"""
        logger.info("Agent 调用完成")
        
        # 移除待移除消息
        current_message_ids = {m.id for m in state.get("messages", []) if m.id}
        message_ids_to_remove = state.get("message_ids_to_remove", [])
        messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in message_ids_to_remove if msg_id in current_message_ids]

        return { "invoke_type": "none", "messages": messages_to_remove, "message_ids_to_remove": [CLEAR] }


agent = Agent()

__all__ = ["agent", "Agent", "ImageMessage"]
