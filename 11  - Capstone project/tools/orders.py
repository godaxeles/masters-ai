# tools/orders.py
from typing import Dict, Any, List, Optional, Tuple
from data.db import safe_select
import logging
import unicodedata

log = logging.getLogger("orders")


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def _russian_stem_token(token: str) -> str:
    """
    Примитивная нормализация русских окончаний: а/я/у/ы/е/и/о/ю/ь.
    Нужна, чтобы 'Ивана' находило 'Иван'.
    """
    if len(token) <= 2:
        return token
    if token[-1] in "аяуыеиоють":
        return token[:-1]
    return token


def _variants_for_like(q: str) -> List[str]:
    """
    Генерируем варианты для LIKE без LOWER (SQLite с кириллицей):
    - как есть, lower, Title, UPPER
    - стемминг последнего символа
    - первый токен (имя без фамилии) и его варианты
    """
    base = _normalize(q)
    if not base:
        return []

    tokens = base.split()
    first = tokens[0]

    stem = _russian_stem_token(base)
    stem_first = _russian_stem_token(first)

    cand = {
        base,
        base.lower(),
        base.title(),
        base.upper(),
        first,
        first.lower(),
        first.title(),
        first.upper(),
        stem,
        stem_first,
        stem.title(),
        stem.upper(),
        stem_first.title(),
        stem_first.upper(),
    }

    return [v for v in (s.strip() for s in cand) if v]


def _build_filters(
    customer_contains: Optional[str],
    city_equals: Optional[str],
    product_contains: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[str, tuple]:
    """
    Формирует безопасный WHERE и параметры.
    ВАЖНО: не используем LOWER/NOCASE — только набор LIKE по вариантам.
    """
    where: List[str] = []
    params: List[Any] = []

    # Клиент (имя/название): набор вариантов
    if customer_contains:
        variants = _variants_for_like(customer_contains)
        if variants:
            where.append("(" + " OR ".join(["customer LIKE ?"] * len(variants)) + ")")
            params.extend([f"%{v}%" for v in variants])

    # Город: допускаем разные регистры
    if city_equals:
        c = _normalize(city_equals)
        variants = list({c, c.title(), c.upper()})
        where.append("(" + " OR ".join(["city LIKE ?"] * len(variants)) + ")")
        params.extend(variants)

    # Товар: частичное совпадение по разным регистрам
    if product_contains:
        p = _normalize(product_contains)
        variants = list({p, p.lower(), p.title(), p.upper()})
        where.append("(" + " OR ".join(["product LIKE ?"] * len(variants)) + ")")
        params.extend([f"%{v}%" for v in variants])

    # Даты
    if date_from:
        where.append("order_date >= ?")
        params.append(_normalize(date_from))
    if date_to:
        where.append("order_date <= ?")
        params.append(_normalize(date_to))

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    return clause, tuple(params)


def fetch_orders(
    customer_contains: Optional[str] = None,
    city_equals: Optional[str] = None,
    product_contains: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Безопасная выборка заказов. Модель НЕ передаёт сырой SQL.
    Если фильтров нет — вернём последние заказы (LIMIT).
    """
    clause, params = _build_filters(
        customer_contains=customer_contains,
        city_equals=city_equals,
        product_contains=product_contains,
        date_from=date_from,
        date_to=date_to,
    )

    sql = (
        "SELECT id, order_date, customer, city, product, quantity, price, "
        "(quantity*price) AS total "
        f"FROM orders {clause} "
        "ORDER BY order_date DESC "
        "LIMIT ?"
    )
    full_params = params + (limit,)

    # Лог запроса инструмента (дополнительно к логам data.db)
    log.info("ORDERS TOOL SQL> %s | params=%s", sql, full_params)

    rows = safe_select(sql, full_params)
    return {"ok": True, "rows": rows[:limit], "debug": {"sql": sql, "params": full_params}}


def summarize_rows(rows: List[dict], max_rows: int = 12) -> str:
    """
    Компактная текстовая выжимка результатов для передачи в LLM.
    """
    rows = rows[:max_rows]
    if not rows:
        return "Нет данных."
    return "\n".join(
        f"#{r['id']} {r['order_date']} {r['customer']} ({r['city']}) — "
        f"{r['product']} x{r['quantity']} = {r['total']:.2f}"
        for r in rows
    )
