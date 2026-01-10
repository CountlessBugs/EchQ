"""主程序入口模块

负责初始化各个组件并启动 Napcat 消息监听服务。
处理来自 Napcat 的消息并通过 Agent 生成回复。
"""

import asyncio
import json
from typing import Any, AsyncIterator
import logging

from config.config import Config
from napcat.napcat import napcat_client, napcat_listener
from napcat.message_formatter import NapcatMessage
from agent.agent import agent
from agent.tools.image_generation_tools import generate_image_tool
from agent.tools.sound_tools import play_sound_tool
from utils.image_utils import image_utils

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("echq.log"), # 保存到文件
        logging.StreamHandler()          # 同时输出到控制台
    ]
)


# === 程序入口与主循环 ===

async def main() -> None:
    """主函数入口"""
    try:
        # 初始化所有组件
        initialize_components()
        
        # 启动监听器
        await napcat_listener.start()

        print("==================================================")
        print("=            INITIALIZATION  COMPLETE            =")
        print("=       Agent 已启动, 按下 Ctrl+C 退出程序       =")
        print("==================================================")
        
        # 保持程序运行，直到收到退出信号
        stop_event = asyncio.Event()
        await stop_event.wait()

    except Exception as e:
        print(f"❌ 不好啦! 程序运行出错: {e}")
    finally:
        # 资源清理
        await cleanup()
        print("Agent 睡着啦! 再见👋🤖")

# === 初始化函数 ===

def initialize_components() -> None:
    """初始化所有组件
    
    按顺序初始化：
    1. Agent 记忆管理器
    2. Agent 核心
    3. Napcat HTTP 客户端
    4. Napcat WebSocket 监听器
    """
    # 初始化 Agent
    # 在此处添加所需的工具
    tools = [generate_image_tool, play_sound_tool]
    agent.initialize(
        llm_model=Config.LLM_MODEL,
        llm_temperature=Config.LLM_TEMPERATURE,
        llm_prompt=Config.LLM_PROMPT,
        token_limit=Config.AMEM_TOKEN_LIMIT,
        llm_model_provider=Config.LLM_MODEL_PROVIDER,
        enable_vision=Config.LLM_ENABLE_VISION,
        tools=tools
    )
    
    # 初始化 Napcat HTTP 客户端
    napcat_client.initialize(Config.NAPCAT_HTTP_URL)
    
    # 初始化 Napcat WebSocket 监听器
    napcat_listener.initialize(
        ws_url=Config.NAPCAT_WS_URL,
        on_message_callback=handle_napcat_message,
        filter_heartbeat=Config.FILTER_WS_HEARTBEAT,
        print_messages=Config.PRINT_WS_MESSAGES
    )
    
    print("✓ 所有组件初始化完成")

# === 清理函数 ===

async def cleanup() -> None:
    """清理资源并关闭连接"""
    print("🧹 正在清理资源...")
    await napcat_client.close()
    await napcat_listener.stop()
    await image_utils.close()
    print("✓ 资源清理完成")

# === 消息处理 ===

async def handle_napcat_message(message: str) -> None:
    """处理 Napcat 消息并生成回复
    
    根据消息类型 (私聊/群聊) 调用 Agent 处理消息,
    并将生成的回复发送回 Napcat
    
    Args:
        message: JSON 格式的消息字符串
    """
    # 解析消息数据
    try:
        message_data: dict[str, Any] = json.loads(message)
    except json.JSONDecodeError as e:
        print(f"⚠️ 不太妙: 消息解析失败: {e}")
        return
    
    if not isinstance(message_data, dict):
        return

    # 根据消息类型处理
    post_type: str = message_data.get("post_type", "")
    
    if post_type == "message":
        await _reply_to_message(message_data)
    elif post_type == "meta_event":
        _handle_meta_event(message_data)
    elif post_type == "notice":
        _handle_notice(message_data)

async def _reply_to_message(message_data: dict[str, Any]) -> None:
    """回复 Napcat 收到的消息
    
    Args:
        message_data: 消息数据字典
    """
    message = NapcatMessage(message_data)
    
    if Config.ENABLE_COMMANDS and message.is_command:
        print(f"⚡ 收到指令: {message.text_content}")
        # 处理指令
        await _handle_command(message)
    else:
        if message.content_type == "text":
            # 打印收到的消息
            print(f"📨 收到消息: {message.message_text}")
            
            # 发送消息给 Agent 并获取回复流
            chunks = agent.invoke("user_message", message.message_text)
        elif message.content_type == "image":
            # 处理图片消息
            b64 = await image_utils.get_remote_image_b64(message.url, 5, 256, 70)
            if b64:
                image_msg = {
                        "text": message.message_text,
                        "images": [b64]
                    }
                print(f"📨 收到图片消息: {message.message_text} [image]{message.url}")
                
                # 发送图片消息给 Agent 并获取回复流
                chunks = agent.invoke("user_message", image_msg)
            else:
                print(f"📨 收到消息: {message.message_text}")

                # 发送消息给 Agent 并获取回复流
                chunks = agent.invoke("user_message", message.message_text)
        
        response_stream = agent.process_chunks(chunks)
        # 逐块发送回复
        async for chunk in response_stream:
            if isinstance(chunk, dict):
                await _send_reply(chunk.get("type", "text"), chunk.get("content", ""), message)
            elif isinstance(chunk, str):
                await _send_reply("text", chunk, message)

async def _handle_command(message: NapcatMessage) -> None:
    """处理收到的指令消息
    
    Args:
        message: 消息对象
    """
    command_echo: str = ""
    match message.command_name:
        case "help":
            if message.command_args:
                command_echo = "❌ 指令 /help 不接受任何参数"
            else:
                command_echo = (
                    "可用指令:\n"
                    "/help - 显示此帮助信息\n"
                    "/context - 查看当前上下文记忆\n"
                    "/token - 查看当前上下文记忆的 token 数量"
                )
        case "context":
            if message.command_args:
                command_echo = "❌ 指令 /context 不接受任何参数"
            else:
                command_echo = "当前上下文记忆(不包含系统提示词):"
                # FIXME: 上下文过长时无法在一条 QQ 中发送，需要添加翻页功能
                for msg in agent.context:
                    # 跳过第一条系统提示词
                    if msg.type == "system" and msg == agent.context[0]:
                        continue
                    if isinstance(msg.content, list):
                        command_echo += f"\n[{msg.type}] "
                        for part in msg.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                command_echo += part.get("text", "")
                            elif isinstance(part, dict) and part.get("type") == "image_url":
                                command_echo += "[image]"
                            else:
                                command_echo += str(part)
                    else:
                        command_echo += f"\n[{msg.type}] {msg.content}"
        case "token":
            if message.command_args:
                command_echo = "❌ 指令 /token 不接受任何参数"
            else:
                command_echo = f"当前上下文记忆的 token 数量: {agent.token_usage}"
        case _:
            command_echo = "🤔 未知指令, 发送 /help 获取帮助"
    
    if command_echo:
        await _send_reply("text", command_echo, message)

async def _send_reply(type: str, content: str, reply_message: NapcatMessage) -> None:
    """根据消息类型发送回复
    
    Args:
        content: 要发送的内容
        message: 原始消息对象
    """
    message_list = []
    if type == "text":
        message_list = [{"type": "text", "data": {"text": content}}]
    elif type in ["image", "record", "file"]:
        message_list = [{"type": type, "data": {"file": content}}]

    if reply_message.message_type == "private":
        await napcat_client.send_message(message_list, reply_message.sender_id, is_group=False)
    elif reply_message.message_type == "group":
        await napcat_client.send_message(message_list, reply_message.group_id, is_group=True)

# === 其他事件处理 ===

def _handle_meta_event(event_data: dict[str, Any]) -> None:
    """处理元事件
    
    Args:
        event_data: 元事件数据字典
    """
    pass


def _handle_notice(notice_data: dict[str, Any]) -> None:
    """处理通知事件
    
    Args:
        notice_data: 通知事件数据字典
    """
    pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
