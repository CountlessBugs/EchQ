import os
import json
from typing import Annotated
import httpx

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..agent_state import ToolCallResult

load_dotenv()


API_KEY = os.getenv("NANO_BANANA_API_KEY")
API_URL = os.getenv("NANO_BANANA_API_URL")
MODEL_NAME = os.getenv("NANO_BANANA_MODEL_NAME", "nano-banana-fast")

@tool("generate_image", parse_docstring=True)
async def generate_image_tool(
    state: Annotated[dict, InjectedState],
    prompt: str
):
    """根据提示词生成图片

    Args:
        prompt: 详细描述你想要生成的画面内容
    """
    # 准备请求参数
    url = API_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    # TODO: 根据实际 API 需求调整负载内容
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "aspectRatio": "auto"
    }

    final_results = []
    error_msg = None

    # 发起异步流式请求
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    status_error = await response.aread()
                    return f"API 错误 (状态码 {response.status_code}): {status_error.decode()}"

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # 兼容 SSE 格式 (data: {...}) 或 纯 JSON 块格式
                    clean_line = line.replace("data: ", "").strip()
                    try:
                        data = json.loads(clean_line)

                        if data.get("status") == "succeeded":
                            final_results = data.get("results", [])
                        elif data.get("status") == "failed":
                            error_msg = data.get("failure_reason") or data.get("error")
                            
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        return f"网络调用异常: {str(e)}"

    # 处理最终状态并构造 Command
    if error_msg:
        return f"图片生成失败: {error_msg}"

    if not final_results:
        return "图片生成已完成，但未返回结果 URL"

    # 构造反馈给 LLM 的文本
    agent_feedback = "图片生成成功！"

    # 获取 tool_call_id
    last_message = state["messages"][-1]
    current_tool_call = next(
        (tc for tc in last_message.tool_calls if tc["name"] == "generate_image"),
        None
    )
    if not current_tool_call:
        return "无法获取 tool_call_id"
    tid = current_tool_call["id"]

    # 构造发送到 QQ 的结构化结果列表 (ToolCallResult)
    structured_results: list[ToolCallResult] = [
        {
            "tool_name": "generate_image",
            "id": f"{tid}_{index}",
            "type": "image",
            "content": r["url"]
        }
        for index, r in enumerate(final_results) if "url" in r
    ]

    # 返回 Command 更新状态
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=agent_feedback,
                    tool_call_id=tid
                )
            ],
            "tool_call_results": structured_results 
        }
    )


__all__ = ["generate_image_tool"]
