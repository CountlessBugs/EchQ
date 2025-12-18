"""主程序入口模块

负责初始化各个组件并启动 Napcat 消息监听服务。
处理来自 Napcat 的消息并通过 Agent 生成回复。
"""

import json
from typing import Any, Dict, Generator

from config.config import Config
from agent.agent import agent
from agent.agent_memory import AgentMemory
from napcat.napcat import napcat_client, napcat_listener
from napcat.message_formatter import NapcatMessage


# === 程序入口与主循环 ===

def main() -> None:
    """主函数入口"""
    try:
        # 初始化所有组件
        initialize_components()
        
        # 启动监听器
        napcat_listener.start()

        # 运行主循环
        run_main_loop()
        
    except Exception as e:
        print(f'❌ 不好啦! 程序运行出错: {e}')
        raise
    finally:
        # 确保资源被正确清理
        cleanup()
        print('Agent 睡着啦! 再见👋🤖')

def run_main_loop() -> None:
    """运行主循环
    
    保持主线程运行，直到用户输入 'exit' 命令退出。
    """
    print('==================================================')
    print('=            INITIALIZATION  COMPLETE            =')
    print('=       Agent 已启动, 输入 "exit" 退出程序       =')
    print('==================================================')
    
    try:
        while True:
            user_input: str = input()
            match user_input.strip().lower():
                case 'help':
                    print('可用命令:')
                    print('  help - 显示此帮助信息')
                    print('  exit - 退出程序')
                    print('  context - 查看当前上下文记忆')
                    print('  token - 查看当前上下文记忆的 token 数量')
                case 'exit':
                    print('\n👋 正在退出...')
                    break
                case 'context':
                    print('当前上下文记忆:')
                    for msg in agent.memory.context_memory:
                        print(f'[{msg['role']}] {msg['content']}')
                case 'token':
                    print(f'当前上下文记忆的 token 数量: {agent.memory.current_token_usage}')
                case _:
                    print('🤔 未知命令, 输入help获取帮助')

    except KeyboardInterrupt:
        print('\n\n⚠️ 检测到中断信号，正在退出...')
    except EOFError:
        print('\n\n⚠️ 输入流已关闭，正在退出...')

# === 初始化函数 ===

def initialize_components() -> None:
    """初始化所有组件
    
    按顺序初始化：
    1. Agent 记忆管理器
    2. Agent 核心
    3. Napcat HTTP 客户端
    4. Napcat WebSocket 监听器
    """
    # 初始化 Agent 记忆管理器
    agent_memory = AgentMemory(
        token_limit=Config.AMEM_TOKEN_LIMIT,
        expected_token_usage=Config.AMEM_EXPECTED_TOKEN_USAGE,
        enable_cache_management=Config.AMEM_ENABLE_CACHE_MANAGEMENT,
        cache_expiry_seconds=Config.AMEM_CACHE_EXPIRY_SECONDS,
        cache_price_ratio=Config.AMEM_CACHE_PRICE_RATIO
    )
    
    # 初始化 Agent
    agent.initialize(
        memory=agent_memory,
        llm_api_url=Config.LLM_API_URL,
        llm_api_key=Config.LLM_API_KEY,
        llm_model=Config.LLM_MODEL,
        llm_temperature=Config.LLM_TEMPERATURE,
        llm_prompt=Config.LLM_PROMPT,
        can_see_datetime=Config.AGENT_CAN_SEE_DATETIME
    )
    
    # 初始化 Napcat HTTP 客户端
    napcat_client.initialize(Config.NAPCAT_HTTP_URL)
    
    # 初始化 Napcat WebSocket 监听器
    napcat_listener.initialize(
        ws_url=Config.NAPCAT_WS_URL,
        on_message_callback=reply_to_napcat_message,
        filter_heartbeat=Config.FILTER_WS_HEARTBEAT
    )
    
    print('✓ 所有组件初始化完成')

# === 清理函数 ===

def cleanup() -> None:
    """清理资源并关闭连接"""
    print('🧹 正在清理资源...')
    napcat_listener.stop()
    print('✓ 资源清理完成')

# === 消息处理 ===

def reply_to_napcat_message(message: str) -> None:
    """处理 Napcat 消息并生成回复
    
    根据消息类型 (私聊/群聊) 调用 Agent 处理消息,
    并将生成的回复发送回 Napcat
    
    Args:
        message: JSON 格式的消息字符串
    """
    # 解析消息数据
    try:
        message_data: Dict[str, Any] = json.loads(message)
    except json.JSONDecodeError as e:
        print(f"⚠️ 不太妙: 消息解析失败: {e}")
        return
    
    if not isinstance(message_data, dict):
        return

    # 根据消息类型处理
    post_type: str = message_data.get('post_type', '')
    
    if post_type == 'message':
        _handle_message(message_data)
    elif post_type == 'meta_event':
        _handle_meta_event(message_data)
    elif post_type == 'notice':
        _handle_notice(message_data)

def _handle_message(message_data: Dict[str, Any]) -> None:
    """处理收到的消息
    
    Args:
        message_data: 消息数据字典
    """
    message = NapcatMessage(message_data)
    
    if Config.ENABLE_COMMANDS and message.is_command:
        print(f"⚡ 收到指令: {message.text_content}")
        print()
        # 处理指令
        _handle_command(message)
    else:
        # 打印收到的消息
        print(f"📨 收到消息: {message.message_text}")
        print()
        
        # 发送消息给 Agent 并获取回复流
        response_stream: Generator[str, None, None] = agent.process_chunks(
            agent.send_message(message.message_text)
        )
        
        # 逐块发送回复
        for chunk in response_stream:
            _send_reply(chunk, message)

def _handle_command(message: NapcatMessage) -> None:
    """处理收到的指令消息
    
    Args:
        message: 消息对象
    """
    command_echo: str = ''
    match message.command_name:
        case 'help':
            if message.command_args:
                command_echo = '❌ 指令 /help 不接受任何参数'
            else:
                command_echo = (
                    '可用指令:\n'
                    '/help - 显示此帮助信息\n'
                    '/context - 查看当前上下文记忆\n'
                    '/token - 查看当前上下文记忆的 token 数量'
                )
        case 'context':
            if message.command_args:
                command_echo = '❌ 指令 /context 不接受任何参数'
            else:
                command_echo = '当前上下文记忆:'
                for msg in agent.memory.context_memory:
                    command_echo += f'\n[{msg['role']}] {msg['content']}'
        case 'token':
            if message.command_args:
                command_echo = '❌ 指令 /token 不接受任何参数'
            else:
                command_echo = f'当前上下文记忆的 token 数量: {agent.memory.current_token_usage}'
        case _:
            command_echo = '🤔 未知指令, 发送 /help 获取帮助'
    
    if command_echo:
        _send_reply(command_echo, message)

def _send_reply(content: str, message: NapcatMessage) -> None:
    """根据消息类型发送回复
    
    Args:
        content: 要发送的内容
        message: 原始消息对象
    """
    if message.message_type == 'private':
        napcat_client.send_text_message(content, message.sender_id)
    elif message.message_type == 'group':
        napcat_client.send_text_message(
            content,
            message.group_id,
            is_group=True
        )

# === 其他事件处理 ===

def _handle_meta_event(event_data: Dict[str, Any]) -> None:
    """处理元事件
    
    Args:
        event_data: 元事件数据字典
    """
    print()  # 终端输出添加换行


def _handle_notice(notice_data: Dict[str, Any]) -> None:
    """处理通知事件
    
    Args:
        notice_data: 通知事件数据字典
    """
    print()  # 终端输出添加换行


if __name__ == '__main__':
    main()
