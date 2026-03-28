import json
from typing import Any, Callable, Dict, List


def get_weather(city_name: str) -> str:
    """模拟查询天气信息，返回给大模型的必须是 JSON 字符串。"""
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

        mock_weather_data = {
            "北京": {"weather": "晴", "temperature": "22C", "humidity": "35%"},
            "重庆": {"weather": "多云", "temperature": "26C", "humidity": "68%"},
            "上海": {"weather": "小雨", "temperature": "20C", "humidity": "82%"},
        }
        weather_info = mock_weather_data.get(
            normalized_city,
            {"weather": "未知", "temperature": "N/A", "humidity": "N/A"},
        )

        return json.dumps(
            {
                "status": "success",
                "city_name": normalized_city,
                "data": weather_info,
            },
            ensure_ascii=False,
        )
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
    """模拟查询学生课表，返回给大模型的必须是 JSON 字符串。"""
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

        mock_schedule_data = {
            "20250001": [
                {"weekday": "周一", "course_name": "高等数学", "time": "08:00-09:40", "location": "第一教学楼 A101"},
                {"weekday": "周三", "course_name": "Python 程序设计", "time": "10:00-11:40", "location": "实验楼 B203"},
            ],
            "20250002": [
                {"weekday": "周二", "course_name": "大学英语", "time": "08:00-09:40", "location": "第二教学楼 C305"},
                {"weekday": "周四", "course_name": "数据结构", "time": "14:00-15:40", "location": "实验楼 A402"},
            ],
        }
        schedule_info = mock_schedule_data.get(normalized_student_id, [])

        return json.dumps(
            {
                "status": "success",
                "student_id": normalized_student_id,
                "data": schedule_info,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "课表接口超时，请安抚用户。",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )


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
