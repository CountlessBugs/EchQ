from typing import Annotated

from pydantic import Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..agent_state import ToolCallResult
from EchQ.config.paths import Paths


@tool("play_sound", parse_docstring=True)
def play_sound_tool(
    state: Annotated[dict, InjectedState],
    file_name: str
):
    """播放音效工具函数

        通过指定的文件名播放本地音效

    Args:
        file_name: 本地音效文件名
    """
    sound_file = Paths.SOUNDS / file_name
    if not sound_file.exists():
        return f"音效文件 {file_name} 不存在"
    
    # 获取 tool_call_id
    last_message = state["messages"][-1]
    current_tool_call = next(
        (tc for tc in last_message.tool_calls if tc["name"] == "play_sound"),
        None
    )
    if not current_tool_call:
        return "无法获取 tool_call_id"
    tid = current_tool_call["id"]

    # 构造发送到 QQ 的结构化结果列表 (ToolCallResult)
    structured_results: list[ToolCallResult] = [
        {
            "tool_name": "play_sound",
            "id": f"{tid}",
            "type": "file",
            "content": str(sound_file)
        }
    ]

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"播放音效：{file_name}",
                    tool_call_id=tid
                )
            ],
            "tool_call_results": structured_results
        }
    )


__all__ = ["play_sound_tool"]
