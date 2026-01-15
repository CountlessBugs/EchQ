from typing import Optional
import logging
import time
import datetime
import math

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.paths import Paths

logger = logging.getLogger(__name__)


class AgentMemory:
    """Agent 记忆类

    负责管理 Agent 的持久化记忆, 包括向量数据库和图数据库的存取操作等
    向量数据库使用 Chroma 实现

    Attributes:
        base_half_life: 记忆基础半衰期 (单位: 天)
        strength_factor: 回忆强化系数
        importance_curve: 重要性曲线陡峭度
    """
    
    # === 初始化方法 ===

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_half_life: float = 7.0,
        strength_factor: float = 0.5,
        importance_curve: float = 0.3
    ) -> None:
        """初始化记忆组件

        Embedding 使用 OpenAIEmbeddings, OPENAI_API_KEY 需在环境变量中配置
        
        Args:
            embedding_model: 使用的 Embedding 模型名称

            base_half_life: 记忆基础半衰期 (单位: 天), 默认 7.0
                含义: 在该时间后, 未被回忆过的普通记忆 (importance=0) 的分数衰减至 50%
                实际半衰期计算公式: half_life = base_half_life * (1 + recall_count)^strength_factor
                调优建议:
                - 7-14天: 标准配置, 适合大多数场景
                - 3-5天: 快速遗忘, 突出最近和重要记忆
                - 14-30天: 慢速遗忘, 保留更多历史记忆

            strength_factor: 回忆强化系数, 范围 [0.0-1.0], 默认 0.3
                含义: 控制"回忆次数"对延缓遗忘的影响程度
                效果对比 (base_half_life=7.0, recall_count=5):
                - strength_factor=0.3: half_life = 7 * (1+5)^0.3 ≈ 7 * 1.62 ≈ 11.3
                - strength_factor=0.5: half_life = 7 * (1+5)^0.5 ≈ 7 * 2.45 ≈ 17.1
                调优建议:
                - 0.2-0.3: 温和强化, 回忆效果渐进累积
                - 0.4-0.5: 显著强化, 频繁回忆的记忆持久保留
                - 0.0: 禁用强化, 所有记忆按相同速率衰减

            importance_curve: 重要性曲线陡峭度, 范围 [0.5-0.9], 默认 0.7
                含义: 控制 importance 值对分数提升的非线性映射
                计算公式: strength = 1.15 + 0.5 * importance^importance_curve
                效果对比 (importance=0.8):
                - curve=0.5: importance^0.5 = 0.89 -> strength=1.60 -> 提升更激进
                - curve=0.7: importance^0.7 = 0.84 -> strength=1.57 -> 提升适中
                - curve=0.9: importance^0.9 = 0.82 -> strength=1.56 -> 提升保守
                调优建议:
                - 0.7-0.8: 平衡配置
                - 0.5-0.6: 强化高重要性记忆的优势, 拉开差距
                - 0.8-0.9: 所有重要性级别较均衡
        """
        # 初始化 Embeddings
        self._embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=embedding_model)

        # 初始化向量数据库
        self._vector_db: Chroma = Chroma(
            collection_name="episodic_memory",
            embedding_function=self._embeddings,
            persist_directory=Paths.CHROMA_DB.as_posix()
        )

        self.base_half_life: float = base_half_life
        self.strength_factor: float = strength_factor
        self.importance_curve: float = importance_curve

        # 当前上下文中已回忆的记忆ID集合, 避免重复回忆
        self._recalled_memory_ids: set[str] = set()

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

        # TODO: 通过 embeddings 检查内容是否已存在, 避免重复存储
        # 若存在则提升重要性

        docs = [Document(
            page_content=c,
            metadata={
                "type": t,
                "created_at": int(timestamp),
                "importance": imp,
                "last_accessed_at": int(timestamp),
                "recall_count": 0
            },
            id=f"mem_{int(timestamp * 1000)}_{i}"
        ) for i, (c, t, imp) in enumerate(zip(content, type, importance))]

        if docs:
            self._vector_db.add_documents(docs)
            logger.info(f"将 {len(docs)} 条记忆存储到向量数据库, 详情如下:\n{docs}")
        else:
            logger.info("尝试存储的记忆内容列表为空, 操作已忽略")

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
            filter: 过滤条件, 传递给向量数据库
            score_threshold: 最终检索评分阈值, 范围 [0.0-1.0]
                - 建议 0.6 用于刻意回忆 (主动调用回忆工具)
                - 建议 0.8 用于自动联想 (对话中自动触发)
        
        Returns:
            相似记忆文本列表
        """ 
        candidate_k = k * 3  # 多检索一些以便后续筛选

        results = self._vector_db.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filter
        )

        # 重新计算每条记忆的检索分数
        # TODO: 加入随机扰动以增加多样性
        scored_results = []
        current_time = int(time.time())

        for doc, similarity_score in results:
            # 从 metadata 中提取必要信息
            metadata = doc.metadata
            importance = metadata.get('importance', 0.5)  # 默认中等重要性
            last_accessed_at = metadata.get('last_accessed_at', metadata.get('created_at', current_time))
            recall_count = metadata.get('recall_count', 0)
            
            # 计算时间衰减后的检索分数
            retrieval_score = self._calculate_retrieval_score(
                similarity=similarity_score,
                importance=importance,
                last_accessed_at=last_accessed_at,
                recall_count=recall_count,
                current_time=current_time,
            )

            # 保存原始相似度和新的检索分数
            doc.metadata['_similarity_score'] = similarity_score
            doc.metadata['_retrieval_score'] = retrieval_score

            scored_results.append((doc, retrieval_score))

        # 按新的检索分数重新降序排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 筛选记忆并截断到 k 个
        filtered_results = []
        for doc, score in scored_results:
            if score >= score_threshold and doc.id not in self._recalled_memory_ids:
                filtered_results.append((doc, score))
                # 记录到已回忆集合
                self._recalled_memory_ids.add(doc.id)

                if len(filtered_results) >= k:
                    break

        # 日志输出
        if filtered_results:
            logger.info(f"检索到 {len(filtered_results)} 条符合条件的记忆:")
            for i, (doc, score) in enumerate(filtered_results, 1):
                sim_score = doc.metadata.get('_similarity_score', 0)
                importance = doc.metadata.get('importance', 0)
                logger.info(
                    f"  [{i}] 相似度={sim_score:.3f}, 重要性={importance:.2f}, "
                    f"最终分数={score:.3f} | {doc.page_content[:50]}..."
                )
        else:
            logger.info(f"未检索到评分高于 {score_threshold} 的记忆")
        
        # 更新记忆的访问时间和回忆次数
        for doc, _ in filtered_results:
            metadata = doc.metadata
            # 移除临时字段
            metadata.pop('_similarity_score', None)
            metadata.pop('_retrieval_score', None)
            # 更新访问时间和回忆次数
            metadata['last_accessed_at'] = current_time
            metadata['recall_count'] = metadata.get('recall_count', 0) + 1
            # 更新向量数据库中的文档元数据
            self._vector_db.update_document(
                document_id=doc.id,
                document=doc
            )

        return [doc for doc, _ in filtered_results]

    # === 其他方法 ===

    def clear_recalled_memory_ids(self) -> None:
        """清除已回忆记忆ID集合, 以便在新上下文中重新回忆"""
        self._recalled_memory_ids.clear()
        logger.info("已清除已回忆记忆ID集合")

    # === 辅助方法 ===

    def _calculate_retrieval_score(
        self,
        similarity: float,
        importance: float,
        last_accessed_at: int,
        recall_count: int,
        current_time: int
    ) -> float:
        """计算考虑时间衰减的检索分数

        检索分数计算公式:
        1. 时间衰减因子: recency_factor = (1 - importance) * decay + importance
        其中: decay = exp(-ln(2) * days_since_last / half_life)
        
        2. 动态半衰期: half_life = base_half_life * (1 + recall_count)^strength_factor
        
        3. 基础分数: base_score = similarity * recency_factor
        
        4. 最终分数: retrieval_score = base_score^(1/strength)
        其中: strength = 1.15 + 0.5 * importance^importance_curve
        
        Args:
            similarity: embeddings 相似度 [0.0-1.0]
            importance: 记忆重要性 [0.0-1.0]
            last_accessed_at: 上次访问时间的时间戳, 单位为秒
            recall_count: 回忆次数
            current_time: 当前时间的时间戳, 单位为秒
        
        Returns:
            最终检索分数 [0.0-1.0]
        """
        
        # 计算距上次访问的天数
        days_since_last = (current_time - last_accessed_at) / 86400.0  # 转换为天
        days_since_last = max(0, days_since_last)  # 防止负数
        
        # 计算动态半衰期 (回忆次数越多, 遗忘越慢)
        half_life = self.base_half_life * math.pow(1 + recall_count, self.strength_factor)
        
        # 计算时间衰减因子
        k = math.log(2)
        decay = math.exp(-k * days_since_last / half_life)
        
        # 应用重要性权重 (重要记忆衰减慢)
        recency_factor = (1 - importance) * decay + importance
        
        # 计算基础分数
        base_score = similarity * recency_factor
        
        # 非线性重要性提升
        importance_curved = math.pow(importance, self.importance_curve)
        strength = 1.15 + 0.5 * importance_curved
        
        # 指数提升
        retrieval_score = math.pow(base_score, 1 / strength)
        
        return retrieval_score


__all__ = ["AgentMemory"]
