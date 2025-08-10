from typing import List, Dict, Any
import json

from tools.orders import fetch_orders, summarize_rows
from tools.weather import get_weather_for_city
from tools.kpi import get_kpis
from llm.client import get_openai_client

SYSTEM_PROMPT = (
    "Ты — корпоративный ассистент по продажам и логистике. "
    "Твоя зона ответственности: заказы, выручка, KPI, агрегаты по товарам/клиентам/городам, "
    "а также погода для доставки. "
    "Используй ТОЛЬКО инструменты: fetch_orders, get_kpis и get_weather_for_city. "
    "Для KPI всегда используй инструмент get_kpis. "
    "Для заказов используй fetch_orders с параметрами (customer_contains, city_equals, product_contains, date_from, date_to, limit). "
    "Если пользователь просит показать ВСЕ ЗАКАЗЫ — вызови fetch_orders без фильтров с limit=50. "
    "Никогда не придумывай поля и не пиши сырые SQL-условия. "
    "Если запрос вне этой области, ответь: "
    "\"Извините, я могу отвечать только на вопросы по заказам, KPI и погоде.\""
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_orders",
            "description": "Безопасная выборка заказов по параметрам (не принимает сырой SQL). "
                           "Если нужна выдача всех заказов — можно передать пустые параметры и только limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_contains": {"type": "string"},
                    "city_equals": {"type": "string"},
                    "product_contains": {"type": "string"},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kpis",
            "description": "Вернуть бизнес-KPI: суммарная выручка, количество заказов и топ товаров.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_for_city",
            "description": "Получить прогноз погоды для города (Open-Meteo).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False
            }
        }
    }
]

def _exec_tool(tc) -> Dict[str, Any]:
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")

    if name == "fetch_orders":
        res = fetch_orders(
            customer_contains=args.get("customer_contains"),
            city_equals=args.get("city_equals"),
            product_contains=args.get("product_contains"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            limit=int(args.get("limit", 50)),
        )
        content = summarize_rows(res["rows"], 12) if res.get("ok") else f"ERROR: {res}"
        return {"tool_call_id": tc.id, "name": name, "content": content}

    if name == "get_kpis":
        res = get_kpis(int(args.get("top_n", 3)))
        return {"tool_call_id": tc.id, "name": name, "content": json.dumps(res, ensure_ascii=False)}

    if name == "get_weather_for_city":
        res = get_weather_for_city(args.get("city", ""))
        return {"tool_call_id": tc.id, "name": name, "content": json.dumps(res, ensure_ascii=False)}

    return {"tool_call_id": tc.id, "name": name, "content": "ERROR: unknown tool"}

def chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    client, model = get_openai_client()
    if not client:
        return {"type": "text", "content": "(offline) OpenAI клиент не инициализирован."}

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.2
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        tool_outputs = [_exec_tool(tc) for tc in msg.tool_calls]
        extended = messages + [{
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        }]
        for out in tool_outputs:
            extended.append({
                "role": "tool",
                "tool_call_id": out["tool_call_id"],
                "name": out["name"],
                "content": out["content"]
            })
        final = client.chat.completions.create(
            model=model,
            messages=extended,
            temperature=0.2
        ).choices[0].message
        return {"type": "llm", "message": final}

    return {"type": "llm", "message": msg}
