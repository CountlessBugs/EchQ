import asyncio
import json
from typing import Any, Callable, Optional

import httpx
import websockets


class NapcatClient:
    """Napcat HTTP 客户端类
    
    用于通过HTTP API发送QQ消息 (文本, 语音等)
    """
    def __init__(self) -> None:
        """初始化NapcatClient实例"""
        self._client: Optional[httpx.AsyncClient] = None
        self._client_sync: Optional[httpx.Client] = None
        self._base_url: str = ""

    # === 初始化方法 ===

    def initialize(self, base_url: str) -> None:
        """初始化Napcat客户端
        
        Args:
            base_url: Napcat HTTP API的基础URL地址
        """
        self._base_url = base_url.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=self._base_url, 
            timeout=15.0,
            headers={"Content-Type": "application/json"}
        )

        self._client_sync = httpx.Client(
            base_url=self._base_url, 
            timeout=15.0,
            headers={"Content-Type": "application/json"}
        )

    async def close(self) -> None:
        """关闭客户端，释放资源"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # === 发送消息方法 ===

    async def send_message(
        self,
        message: list[dict[str, Any]],
        receiver: str,
        is_group: bool = False
    ) -> dict[str, Any]:
        """发送QQ消息
        
        Args:
            message: 要发送的消息内容列表
            receiver: 消息接收者的QQ号或群号
            is_group: 消息接收者是否为群聊, 默认为False
        
        Returns:
            Napcat API的响应结果字典
        """
        if not self._client:
            raise RuntimeError("NapcatClient 未初始化，请先调用 initialize()")

        payload: dict[str, Any] = {"message": message}
        if is_group:
            endpoint = "/send_group_msg"
            payload["group_id"] = receiver
        else:
            endpoint = "/send_private_msg"
            payload["user_id"] = receiver
        
        try:
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Napcat 发送失败: {e}")
            return {"status": "failed", "error": str(e)}

    # === 获取消息方法 ===

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """获取指定ID的消息详情
        
        Args:
            message_id: 要获取的消息ID
        
        Returns:
            消息详情字典
        """
        if not self._client:
            raise RuntimeError("NapcatClient 未初始化，请先调用 initialize()")
        
        try:
            payload = {"message_id": message_id}
            response = await self._client.post("/get_msg", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # 捕获 HTTP 错误
            error_msg = f"HTTP错误: {e.response.status_code} - {e.response.text}"
            print(f"❌ 获取消息失败: {error_msg}")
            return {"status": "failed", "error": error_msg}
        except Exception as e:
            print(f"❌ 获取消息失败: {e}")
            return {"status": "failed", "error": str(e)}

    def get_message_sync(self, message_id: str) -> dict[str, Any]:
        """同步获取指定ID的消息详情
        
        Args:
            message_id: 要获取的消息ID
        
        Returns:
            消息详情字典
        """
        if not self._client_sync:
            raise RuntimeError("NapcatClient 未初始化，请先调用 initialize()")
        
        try:
            payload = {"message_id": message_id}
            response = self._client_sync.post("/get_msg", json=payload)
            response.raise_for_status()
            return response.json().get("data", {})
        except httpx.HTTPStatusError as e:
            # 捕获 HTTP 错误
            error_msg = f"HTTP错误: {e.response.status_code} - {e.response.text}"
            print(f"❌ 获取消息失败: {error_msg}")
            return {"status": "failed", "error": error_msg}
        except Exception as e:
            print(f"❌ 获取消息失败: {e}")
            return {"status": "failed", "error": str(e)}


class NapcatListener:
    """Napcat WebSocket 监听器类
    
    基于 asyncio 和 websockets 实现，用于监听 Napcat 事件流
    采用非阻塞设计，支持在单个线程内与其他异步任务并发运行
    
    Attributes:
        on_message_callback (Optional[Callable[[str], None]]): 接收到消息时的回调函数
        filter_heartbeat (bool): 是否过滤心跳消息
    """
    def __init__(self) -> None:
        """初始化NapcatListener实例"""
        self._ws_url: str = ""
        self.on_message_callback: Optional[Callable[[str], None]] = None
        self.filter_heartbeat: bool = True
        self.print_messages: bool = False
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    # === 初始化方法 ===

    def initialize(
        self,
        ws_url: str,
        on_message_callback: Optional[Callable[[str], None]] = None,
        filter_heartbeat: bool = True,
        print_messages: bool = False
    ) -> None:
        """初始化Napcat监听器
        
        Args:
            ws_url: Napcat WebSocket服务的URL地址
            on_message_callback: 接收到消息时的回调函数, 默认为None
            filter_heartbeat: 是否过滤心跳消息, 默认为True
            print_messages: 是否打印接收消息日志, 默认为False
        """
        self._ws_url = ws_url
        self.on_message_callback = on_message_callback
        self.filter_heartbeat = filter_heartbeat
        self.print_messages = print_messages
        self._running = False
        self._task = None

    # === 监听器启动与停止方法 ===

    async def start(self) -> None:
        """启动监听器"""
        if self._running:
            print("Napcat监听器已在运行中")
            return
        
        self._task = asyncio.create_task(self._run())
        self._running = True
        print("Napcat监听器已启动")

    async def stop(self) -> None:
        """停止监听器"""
        if not self._running:
            print("Napcat监听器未在运行中")
            return
        
        # 发送取消信号，_run 中的 await 处会抛出 CancelledError
        self._task.cancel()
        try:
            await self._task # 等待任务优雅退出
        except asyncio.CancelledError:
            pass

        print("Napcat监听器已停止运行. Nap cat went for a nap~ 😸💤")

    # === 私有方法 ===

    async def _run(self) -> None:
        """运行监听器主循环"""
        try:
            # 建立连接
            async with websockets.connect(self._ws_url) as ws:
                print("✓ 已连接到Napcat WebSocket! 好耶!")
                
                # 接收消息
                async for message in ws:
                    asyncio.create_task(self._on_message(message))
                    
        # 处理连接异常
        except ConnectionRefusedError:
            print("❌ 不好啦! 连接被拒绝: NapCat WebSocket 服务未运行或端口不正确")
            print(f"   请检查: {self._ws_url}")
        except (asyncio.TimeoutError, OSError) as e:
            print(f"❌ 不好啦! 连接超时或错误: {e}")
        except asyncio.CancelledError:
            # 任务被取消时的正常退出
            print("Napcat Websocket已关闭")
            raise
        except Exception as e:
            print(f"❌ Napcat监听器运行时发生错误: {e}")
        finally:
            self._running = False

    async def _on_message(self, message: str) -> None:
        """回调方法: 接收到消息
        
        Args:
            message: 接收到的消息字符串
        """
        try:
            message_data: dict[str, Any] = json.loads(message)
            
            # 过滤心跳消息
            if (isinstance(message_data, dict) and self.filter_heartbeat
                and message_data.get("post_type") == "meta_event"
                and message_data.get("meta_event_type") == "heartbeat"):
                return
            
            if self.print_messages:
                print(f"Napcat 监听器收到消息: {message}")

            if self.on_message_callback:
                if self.on_message_callback:
                    # 检查回调是否是异步函数，如果是则 await，否则直接调用
                    if asyncio.iscoroutinefunction(self.on_message_callback):
                        await self.on_message_callback(message)
                    else:
                        self.on_message_callback(message)
        except json.JSONDecodeError:
            print(f"消息解析失败: {message}")
        except Exception as e:
            print(f"处理消息时发生错误: {e}")


# 全局Napcat客户端和监听器实例
napcat_client = NapcatClient()
napcat_listener = NapcatListener()

__all__ = ["napcat_client", "napcat_listener"]
