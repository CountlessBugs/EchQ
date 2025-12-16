import json
from pathlib import Path
import time
import threading
from typing import Optional, Literal, Dict, List, Any

from Echo.config.config_templete import CONFIG_DIR

# 定义消息角色类型
MessageRole = Literal['user', 'assistant', 'system', 'tool']


class AgentMemory:
    """管理上下文、长期记忆和对话历史记录的类
    
    该类负责管理智能体的对话记忆，包括上下文管理、自动清理、
    缓存管理以及对话历史的持久化存储。
    
    Attributes:
        context_memory: 上下文记忆列表，存储当前会话的消息
        auto_clear_context: 是否自动清除上下文记忆
        token_limit: 上下文记忆的最大 token 数限制
        expected_token_usage: 每次对话消耗 token 的期望值, 缓存 token 会按照价格比折合
        enable_cache_management: 是否启用缓存管理
        cache_expiry_seconds: 上下文记忆缓存过期时间(秒), 计时时间为实际过期时间减去5秒
        cache_price_ratio: 上下文记忆缓存价格比率, 用于计算等效 token 使用量
        current_token_usage: 当前一次对话消耗的 token 数
    """
    def __init__(
        self,
        auto_clear_context: bool = True,
        token_limit: int = 64000,
        expected_token_usage: int = 16000,
        enable_cache_management: bool = False,
        cache_expiry_seconds: int = 300,
        cache_price_ratio: float = 0.5
    ) -> None:
        """初始化智能体记忆
        
        Args:
            auto_clear_context: 是否自动清除上下文记忆
            token_limit: 上下文记忆的最大 token 数限制
            expected_token_usage: 每次对话期望的 token 消耗量
            enable_cache_management: 是否启用缓存管理功能
            cache_expiry_seconds: 缓存过期时间（秒）
            cache_price_ratio: 缓存价格比率，范围 (0, 1]
            
        Raises:
            ValueError: 当参数不符合要求时抛出
        """
        # 参数验证
        if token_limit <= 0:
            raise ValueError('token_limit 必须为正整数')
        if expected_token_usage <= 0:
            raise ValueError('expected_token_usage 必须为正整数')
        if cache_expiry_seconds <= 5:
            raise ValueError('cache_expiry_seconds 必须大于 5 秒')
        if cache_price_ratio <= 0 or cache_price_ratio > 1:
            raise ValueError('cache_price_ratio 必须在 (0, 1] 范围内')

        # 初始化属性
        self.context_memory: List[Dict[str, str]] = []
        self.auto_clear_context: bool = auto_clear_context
        self.token_limit: int = token_limit
        self.expected_token_usage: int = expected_token_usage
        self.enable_cache_management: bool = enable_cache_management
        self.cache_expiry_seconds: int = cache_expiry_seconds
        self.cache_price_ratio: float = cache_price_ratio
        
        # 私有属性
        self._current_token_usage: int = 0
        self._cache_expire_timer: Optional[threading.Timer] = None

    def add_message(
        self,
        message: Optional[Dict[str, str]] = None,
        /,
        *,
        role: Optional[MessageRole] = None,
        content: Optional[str] = None
    ) -> None:
        """添加消息到上下文记忆，并保存至本地对话历史文件
        
        支持两种调用方式：
        1. 传入完整的 message 字典
        2. 分别指定 role 和 content 参数
        
        Args:
            message: 完整消息字典，包含 'role' 和 'content' 键
            role: 消息角色，可选值为 'user', 'assistant', 'system' 或 'tool'
            content: 消息内容
            
        Raises:
            ValueError: 当消息格式不正确或角色无效时抛出
        """
        # 处理消息参数
        if message:
            role = message.get('role')  # type: ignore
            content = message.get('content')
        
        # 验证消息完整性
        if role is None or content is None:
            raise ValueError('必须提供完整的消息')
        if role not in ['user', 'assistant', 'system', 'tool']:
            raise ValueError("角色必须是 'user'、'assistant'、'system' 或 'tool'")
        
        # 添加到上下文记忆
        self.context_memory.append({'role': role, 'content': content})

        # 保存到对话历史文件
        timestamp: int = int(time.time())
        self._save_to_archive(role, content, timestamp)

        # 刷新缓存过期定时器（非 assistant 消息时）
        if role != 'assistant' and self.enable_cache_management:
            self._refresh_cache_timer()

    def clear_context_memory(self) -> None:
        """清除上下文记忆并重置相关状态"""
        self.context_memory = []
        self._current_token_usage = 0
        
        # 停止缓存过期定时器
        if self._cache_expire_timer:
            self._cache_expire_timer.cancel()
            self._cache_expire_timer = None

    def summarize_context(self) -> None:
        """总结上下文记忆以减少 token 使用
        
        调用外部 agent 生成上下文摘要, 然后清除现有上下文,
        并将摘要作为系统消息添加回上下文中.
        """
        from .agent import agent
        
        summary: str = agent.get_context_summary()
        self.clear_context_memory()
        self.add_message(
            role='system',
            content=f'<context_summary>\n{summary}\n</context_summary>'
        )
        print('✓ 上下文记忆已总结压缩')

    def decide_context_clearing(self, cache_expired: bool = False) -> None:
        """根据配置和当前状态决定是否清理上下文记忆
        
        Args:
            cache_expired: 缓存是否已过期. 如果为 True, 则不应用缓存折扣
        """
        equivalent_token_usage: float = float(self._current_token_usage)

        # 如果缓存未过期且启用了缓存管理，应用缓存价格折扣
        if not cache_expired and self.enable_cache_management:
            equivalent_token_usage *= self.cache_price_ratio

        # 检查是否需要清理上下文
        if (
            equivalent_token_usage > self.expected_token_usage
            or self._current_token_usage > self.token_limit
        ):
            self.summarize_context()

    @property
    def current_token_usage(self) -> int:
        """获取当前上下文记忆的 token 使用情况
        
        Returns:
            当前使用的 token 数量
        """
        return self._current_token_usage

    @current_token_usage.setter
    def current_token_usage(self, value: int) -> None:
        """设置当前上下文记忆的 token 使用情况
        
        设置后会自动检查是否需要清理上下文记忆
        
        Args:
            value: 新的 token 使用量
        """
        self._current_token_usage = value
        if self.auto_clear_context:
            self.decide_context_clearing()

    # 私有辅助方法

    def _save_to_archive(
        self,
        role: str,
        content: str,
        timestamp: int
    ) -> None:
        """将消息保存到本地对话历史文件
        
        Args:
            role: 消息角色
            content: 消息内容
            timestamp: 时间戳
        """
        from pathlib import Path

        # 构建对话历史文件路径
        agent_memory_dir = Path(__file__).parent
        conversation_archive_path = agent_memory_dir / 'conversation_archive.jsonl'

        try:
            with open(conversation_archive_path, 'a', encoding='utf-8') as f:
                archive_entry: Dict[str, Any] = {
                    'role': role,
                    'content': content,
                    'timestamp': timestamp
                }
                json.dump(archive_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"⚠️ 不好啦！保存对话历史时出错: {e}")

    def _refresh_cache_timer(self) -> None:
        """刷新缓存过期定时器"""
        if self._cache_expire_timer:
            self._cache_expire_timer.cancel()
        
        self._cache_expire_timer = threading.Timer(
            self.cache_expiry_seconds - 5,
            self._on_cache_expire
        )
        self._cache_expire_timer.start()

    def _on_cache_expire(self) -> None:
        """缓存过期处理函数"""
        print('⏰ 上下文记忆缓存已过期')
        self.decide_context_clearing(cache_expired=True)
