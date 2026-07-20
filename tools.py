import json
import asyncio
import os
from typing import Any, Callable, Dict, List


def get_weather(city_name: str) -> str:
    """Query a configured MCP weather tool; never fabricate weather data."""
    try:
        normalized_city = (city_name or "").strip()
        if not normalized_city:
            return json.dumps(
                {
                    "status": "error",
                    "message": "城市名称不能为空，请提醒用户重新输入。",
                },
                ensure_ascii=False,
            )

        return _call_mcp_tool("MCP_WEATHER_URL", "MCP_WEATHER_TOOL_NAME", "weather.query", {"city_name": normalized_city})
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "天气接口超时，请安抚用户。",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )


def get_class_schedule(student_id: str) -> str:
    """Query a configured MCP schedule tool; never fabricate schedule data."""
    try:
        normalized_student_id = (student_id or "").strip()
        if not normalized_student_id:
            return json.dumps(
                {
                    "status": "error",
                    "message": "学号不能为空，请提醒用户重新输入。",
                },
                ensure_ascii=False,
            )

        return _call_mcp_tool("MCP_SCHEDULE_URL", "MCP_SCHEDULE_TOOL_NAME", "schedule.query", {"student_id": normalized_student_id})
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "课表接口超时，请安抚用户。",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )


def _call_mcp_tool(endpoint_env: str, name_env: str, default_name: str, arguments: dict[str, str]) -> str:
    endpoint = os.getenv(endpoint_env, "").strip()
    if not endpoint:
        return json.dumps({"status": "error", "code": "TOOL_UNAVAILABLE", "message": "该工具服务尚未配置。"}, ensure_ascii=False)
    from services.mcp_tools import MCPStreamableHTTPTool
    tool = MCPStreamableHTTPTool(endpoint, tool_name=os.getenv(name_env, default_name).strip() or default_name)
    return asyncio.run(tool.call(arguments))


AGENT_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "根据城市名称查询当前天气情况，适用于校园出行、穿衣建议和天气提醒等场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "需要查询天气的城市名称，例如重庆、北京、上海。",
                    }
                },
                "required": ["city_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_class_schedule",
            "description": "根据学生学号查询课程安排，适用于校园助手查询上课时间、地点和课程名称。",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "学生的唯一学号，例如 20250001。",
                    }
                },
                "required": ["student_id"],
                "additionalProperties": False,
            },
        },
    },
]


AVAILABLE_TOOLS_MAP: Dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "get_class_schedule": get_class_schedule,
}
