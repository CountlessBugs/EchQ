from typing import Optional
import time
import logging

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from EchQ.config.paths import Paths

logger = logging.getLogger(__name__)


class AgentMemory:
    """Agent 记忆类

    负责管理 Agent 的持久化记忆, 包括向量数据库和图数据库的存取操作等
    向量数据库使用 Chroma 实现
    """
    
    # === 初始化方法 ===

    def __init__(
        self,
        embeddings_model: str = "text-embedding-3-small"
    ) -> None:
        """初始化记忆组件

        Embedding 使用 OpenAIEmbeddings, OPENAI_API_KEY 需在环境变量中配置
        
        Args:
            embeddings_model: 使用的 Embedding 模型名称
        """
        # 初始化 Embeddings
        self._embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=embeddings_model)

        # 初始化向量数据库
        self._vector_db: Chroma = Chroma(
            collection_name="episodic_memory",
            embedding_function=self._embeddings,
            persist_directory=Paths.CHROMA_DB.as_posix()
        )

    # === 基本记忆存取方法 ===

    def store_memory(
        self,
        content: str | list[str],
        type: str | list[str] = "default",
        importance: float | list[float] = 1.0
    ) -> None:
        """存储记忆片段到向量数据库

        此方法仅负责存储, 不提供要点提取功能, 需要提供处理好的记忆文本内容
        
        Args:
            content: 记忆文本内容
            type: 记忆类型标签, 若为单个字符串则应用于所有内容, 若为列表则与内容一一对应
            importance: 记忆重要性, 决定了记忆优先级和随时间衰减的速度, 若为单个浮点数则应用于所有内容, 若为列表则与内容一一对应 
        """
        # content 统一包装成列表
        if not isinstance(content, list):
            content = [content]

        # 检查 type 与 content 长度是否匹配
        if isinstance(type, list):
            if len(type) != len(content):
                raise ValueError("当 type 为列表时, 其长度必须与 content 列表长度相同")
        else:
            type = [type] * len(content)
        # 检查 importance 与 content 长度是否匹配
        if isinstance(importance, list):
            if len(importance) != len(content):
                raise ValueError("当 importance 为列表时, 其长度必须与 content 列表长度相同")
        else:
            importance = [importance] * len(content)

        timestamp = time.time()

        docs = [Document(
            page_content=c,
            metadata={"type": t, "timestamp": int(timestamp), "importance": imp},
            id=f"mem_{int(timestamp * 1000)}_{i}"
        ) for i, (c, t, imp) in enumerate(zip(content, type, importance))]

        logger.info(f"将 {len(docs)} 条记忆存储到向量数据库, 详情如下:\n{docs}")

        self._vector_db.add_documents(docs)

    def retrieve_similar_memories(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: float = 0.6
    ) -> list[Document]:
        """检索与查询最相似的记忆片段
        
        Args:
            query: 查询文本
            k: 返回的相似记忆数量
            filter: 过滤条件
            score_threshold: 相似度评分阈值, 仅返回评分高于该阈值的记忆
        
        Returns:
            相似记忆文本列表
        """ 
        results = self._vector_db.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )

        # TODO: 加入评分按时间指数衰减等操作

        # 寻找第一个未达到阈值的索引
        cutoff_index = len(results)
        for i, (_, score) in enumerate(results):
            if score < score_threshold: # 相似度低于阈值，说明后续都不合格
                cutoff_index = i
                break

        # 截断列表, 只保留 0 到 cutoff_index-1 的部分
        results = results[:cutoff_index]

        logger.info(f"检索到 {len(results)} 条相似记忆, 详情如下:\n{results}")

        return [doc for doc, _ in results]


__all__ = ["agent_memory"]
