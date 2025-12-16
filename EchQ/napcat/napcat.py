import json
import threading
from typing import Any, Callable, Optional

import requests
import websocket


class NapcatClient:
    """Napcat HTTP客户端类
    
    用于通过HTTP API发送QQ消息 (文本, 语音等)
    
    Attributes:
        base_url: Napcat HTTP API基础URL
    """
    def __init__(self) -> None:
        """初始化NapcatClient实例"""
        self.base_url: str = ''

    def initialize(self, base_url: str) -> None:
        """初始化Napcat客户端
        
        Args:
            base_url: Napcat HTTP API的基础URL地址
        """
        self.base_url = base_url

    def send_message(
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
        payload: dict[str, Any] = {'message': message}

        if is_group:
            endpoint = f'{self.base_url}/send_group_msg'
            payload['group_id'] = receiver
        else:
            endpoint = f'{self.base_url}/send_private_msg'
            payload['user_id'] = receiver
        
        response = requests.post(endpoint, json=payload)
        return response.json()

    def send_text_message(
        self,
        message: str,
        receiver: str,
        is_group: bool = False
    ) -> dict[str, Any]:
        """发送QQ文本消息
        
        Args:
            message: 要发送的文本消息内容
            receiver: 消息接收者的QQ号或群号
            is_group: 消息接收者是否为群聊, 默认为False
        
        Returns:
            Napcat API的响应结果字典
        """
        message_list: list[dict[str, Any]] = [
            {
                'type': 'text',
                'data': {
                    'text': message
                }
            }
        ]
        return self.send_message(message_list, receiver, is_group)

    def send_record_message(
        self,
        file_path: str,
        receiver: str,
        is_group: bool = False
    ) -> dict[str, Any]:
        """发送QQ语音消息
        
        Args:
            file_path: 语音文件路径(本地或网络路径, 格式为 file:// 或 http:// )
            receiver: 消息接收者的QQ号或群号
            is_group: 消息接收者是否为群聊, 默认为False
        
        Returns:
            Napcat API的响应结果字典
        """
        message_list: list[dict[str, Any]] = [
            {
                'type': 'record',
                'data': {
                    'file': file_path
                }
            }
        ]
        return self.send_message(message_list, receiver, is_group)


class NapcatListener:
    """Napcat WebSocket监听器类
    
    用于监听Napcat WebSocket事件并处理接收到的消息
    
    Attributes:
        ws: WebSocket应用实例
        ws_url: Napcat监听器的WebSocket URL
        on_message_callback: 接收到消息时的回调函数
        filter_heartbeat: 是否过滤心跳消息
    """
    def __init__(self) -> None:
        """初始化NapcatListener实例"""
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_url: str = ''
        self.on_message_callback: Optional[Callable[[str], None]] = None
        self.filter_heartbeat: bool = True
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def initialize(
        self,
        ws_url: str,
        on_message_callback: Optional[Callable[[str], None]] = None,
        filter_heartbeat: bool = True
    ) -> None:
        """初始化Napcat监听器
        
        Args:
            ws_url: Napcat WebSocket服务的URL地址
            on_message_callback: 接收到消息时的回调函数, 默认为None
            filter_heartbeat: 是否过滤心跳消息, 默认为True
        """
        self.ws_url = ws_url
        self.on_message_callback = on_message_callback
        self.filter_heartbeat = filter_heartbeat
        self._thread = None
        self._running = False
        
        # 初始化WebSocket应用
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

    def start(self) -> None:
        """启动监听器"""
        if self._running:
            print('Napcat监听器已在运行中')
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print('Napcat监听器已启动')

    def stop(self) -> None:
        """停止监听器"""
        if not self._running:
            print('Napcat监听器未在运行中')
            return
        
        self._running = False
        if self.ws:
            self.ws.close()
        if self._thread:
            self._thread.join()
        print('Napcat监听器已停止运行. Nap cat went for a nap~ 😸💤')

    def _run(self) -> None:
        """运行监听器主循环"""
        self.ws.run_forever()

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """WebSocket连接建立时的回调
        
        Args:
            ws: WebSocket应用实例
        """
        print('✓ 已连接到Napcat WebSocket! 好耶!')

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """接收到消息时的回调
        
        Args:
            ws: WebSocket应用实例
            message: 接收到的消息字符串
        """
        try:
            message_data: dict[str, Any] = json.loads(message)
            
            # 过滤心跳消息
            if (isinstance(message_data, dict) and self.filter_heartbeat
                and message_data.get('post_type') == 'meta_event'
                and message_data.get('meta_event_type') == 'heartbeat'):
                return
            
            print(f'收到消息: {message}')
            if self.on_message_callback:
                self.on_message_callback(message)
        except json.JSONDecodeError:
            print(f'消息解析失败: {message}')

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """WebSocket错误处理回调
        
        Args:
            ws: WebSocket应用实例
            error: 错误对象
        """
        error_str = str(error)
        if '10061' in error_str or 'Connection refused' in error_str:
            print('❌ 不好啦! 连接被拒绝: NapCat WebSocket 服务未运行或端口不正确')
            print(f'   请检查: {self.ws_url}')
        elif '10060' in error_str or 'timed out' in error_str:
            print(f'❌ 不好啦! 连接超时: 无法访问 {self.ws_url}')
        else:
            print(f'❌ 不好啦! WebSocket 错误: {error}')

    def _on_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str]
    ) -> None:
        """WebSocket连接关闭时的回调
        
        Args:
            ws: WebSocket应用实例
            close_status_code: 关闭状态码
            close_msg: 关闭消息
        """
        print('Napcat Websocket已关闭')


# 全局Napcat客户端和监听器实例
napcat_client = NapcatClient()
napcat_listener = NapcatListener()

__all__ = ['napcat_client', 'napcat_listener']
