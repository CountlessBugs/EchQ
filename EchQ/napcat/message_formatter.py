"""Napcat 消息格式化工具模块"""

import json
from typing import Dict, List, Any, Optional, Literal


class NapcatMessage:
    """Napcat 消息数据类

    用于解析和格式化来自 Napcat 的消息数据

    Args:
        message_data (dict): 原始Napcat消息数据字典

    Args:
        message_data: 原始 Napcat 消息数据字典
    
    Attributes:
        message_type (str): 消息类型 ('private' 或 'group')
        raw_message (str): 原始消息字符串
        message_text (str): 附带额外信息的消息纯文本字符串
        text_content (str): 仅消息内容的纯文本字符串
        sender_id (str): 发送者的用户 ID
        sender_nick (str): 发送者的昵称
        group_id (str): 群 ID (如果是群消息)
        group_name (str): 群名称 (如果是群消息)
        reply_receiver_id (str): 待回复者的 ID, 私聊消息为用户 ID, 群消息为群 ID
        is_command (bool): 消息是否为指令
        command_name (Optional[str]): 指令名称 (如果是指令消息)
        command_args (Optional[List[str]]): 指令参数列表 (如果是指令消息)
    """
    def __init__(self, message_data: Dict[str, Any]) -> None:
        self._message_data: Dict[str, Any] = message_data
        self._message_type: Optional[str] = None
        self._raw_message: Optional[str] = None
        self._message_text: Optional[str] = None
        self._text_content: Optional[str] = None
        self._sender_id: Optional[str] = None
        self._sender_nick: Optional[str] = None
        self._group_id: Optional[str] = None
        self._group_name: Optional[str] = None
        self._reply_receiver_id: Optional[str] = None
        self._is_command: Optional[bool] = None
        self._command_name: Optional[str] = None
        self._command_args: Optional[List[str]] = None
        self._face_list: Optional[Dict[str, str]] = None

    # === 属性访问器 ===

    @property
    def message_type(self) -> str:
        """消息类型 ('private' 或 'group')
        
        Returns:
            消息类型字符串
        """
        if self._message_type is None:
            self._message_type = self._message_data.get('message_type', '')
        return self._message_type

    @property
    def raw_message(self) -> str:
        """原始消息字符串
        
        Returns:
            原始消息字符串
        """
        if self._raw_message is None:
            self._raw_message = self._message_data.get('raw_message', '')
        return self._raw_message

    @property
    def message_text(self) -> str:
        """附带额外信息的消息纯文本字符串
        
        根据消息类型添加前缀信息：
        - 私聊消息: [private] 昵称: 内容
        - 群消息: [group] 群名 昵称: 内容
        
        Returns:
            格式化后的消息文本
        """
        if self._message_text is None:
            if self.message_type == 'private':
                self._message_text = f'[private] {self.sender_nick}: {self.text_content}'
            elif self.message_type == 'group':
                self._message_text = f'[group] {self.group_name} {self.sender_nick}: {self.text_content}'
            else:
                self._message_text = self.text_content
        
        return self._message_text

    @property
    def text_content(self) -> str:
        """仅消息内容的纯文本字符串
        
        从消息数组中提取文本和表情内容, 表情用空格分隔
        
        Returns:
            纯文本消息内容
        """
        if self._text_content is None:
            text_parts: List[str] = []
            content_array: List[Dict[str, Any]] = self._message_data.get('message', [])
            for item in content_array:
                item_type: str = item.get('type', '')

                if item_type == 'text':
                    text: str = item.get('data', {}).get('text', '')
                    text_parts.append(text)
                    
                elif item_type == 'face':
                    face_text: Optional[str] = item.get('data', {}).get('raw', {}).get('faceText')
                    if face_text is not None:
                        text_parts.append(face_text)
                    else:
                        # 获取表情 ID
                        face_id: str = item.get('data', {}).get('id', '')
                        text_parts.append('/' + self._get_face_text_by_id(face_id))
                    text_parts.append(' ')  # 表情用空格分隔

            self._text_content = ''.join(text_parts).strip()

        return self._text_content

    @property
    def sender_id(self) -> str:
        """发送者的用户 ID
        
        Returns:
            用户 ID 字符串
        """
        if self._sender_id is None:
            self._sender_id = str(self._message_data.get('sender', {}).get('user_id', ''))
        return self._sender_id

    @property
    def sender_nick(self) -> str:
        """发送者的昵称
        
        Returns:
            用户昵称字符串
        """
        if self._sender_nick is None:
            self._sender_nick = self._message_data.get('sender', {}).get('nickname', '')
        return self._sender_nick
    
    @property
    def group_id(self) -> str:
        """群 ID (如果是群消息)
        
        Returns:
            群 ID 字符串, 非群消息返回空字符串
        """
        if self._group_id is None:
            if self.message_type == 'group':
                self._group_id = str(self._message_data.get('group_id', ''))
            else:
                self._group_id = ''
        return self._group_id
    
    @property
    def group_name(self) -> str:
        """群名称 (如果是群消息)
        
        Returns:
            群名称字符串, 非群消息返回空字符串
        """
        if self._group_name is None:
            if self.message_type == 'group':
                self._group_name = self._message_data.get('group_name', '')
            else:
                self._group_name = ''
        return self._group_name

    @property
    def reply_receiver_id(self) -> str:
        """待回复者的 ID
        
        根据消息类型返回不同的 ID:
        - 私聊消息: 返回发送者用户 ID
        - 群消息: 返回群 ID
        
        Returns:
            接收者 ID 字符串
        """
        if self._reply_receiver_id is None:
            if self.message_type == 'private':
                self._reply_receiver_id = str(self._message_data.get('sender', {}).get('user_id', ''))
            elif self.message_type == 'group':
                self._reply_receiver_id = str(self._message_data.get('group_id', ''))
            else:
                self._reply_receiver_id = ''
        return self._reply_receiver_id

    @property
    def is_command(self) -> bool:
        """消息是否为指令
        
        以 '/' 开头的消息被视为指令
        
        Returns:
            如果是指令则返回 True, 否则返回 False
        """
        if self._is_command is None:
            # FIXME: 群聊中应该检查是否@了机器人自身
            if self._message_data.get('message', []):
                first_item: Dict[str, Any] = self._message_data['message'][0]
                if first_item.get('type') == 'face':
                    self._is_command = False
                else:
                    self._is_command = self.text_content.strip().startswith('/')
            else:
                self._is_command = False
        return self._is_command

    @property
    def command_name(self) -> Optional[str]:
        """指令名称 (如果是指令消息)
        
        Returns:
            指令名称字符串, 如果不是指令消息则返回 None
        """
        if self._command_name is None:
            self._parse_command()
        return self._command_name

    @property
    def command_args(self) -> Optional[List[str]]:
        """指令参数列表 (如果是指令消息)
        
        Returns:
            指令参数列表, 如果不是指令消息则返回 None
        """
        if self._command_args is None:
            self._parse_command()
        return self._command_args

    # === 私有方法 ===

    def _parse_command(self) -> None:
        """解析指令消息, 提取指令名称和参数列表"""
        if self.is_command:
            parts: List[str] = self.text_content.strip().split()
            if parts and len(parts[0]) > 1:
                self._command_name = parts[0][1:]  # 去掉前导 '/'
                if len(parts) > 1:
                    self._command_args = parts[1:]  # 除去指令名称的其余部分
                else:
                    self._command_args = []
            else:
                self._command_name = ''
        else:
            self._command_name = None

    def _get_face_text_by_id(self, face_id: str) -> str:
        """根据表情 ID 获取表情文本表示
        
        Args:
            face_id: 表情 ID 字符串
        
        Returns:
            表情文本表示字符串
        """
        if self._face_list is None:
            # 加载表情列表 JSON 文件
            from pathlib import Path

            face_list_path: Path = Path(__file__).parent / 'face_list.json'
            with open(face_list_path, 'r', encoding='utf-8') as f:
                self._face_list: Dict[str, str] = json.load(f)

        return self._face_list.get(face_id, '表情')


__all__ = ['NapcatMessage']
