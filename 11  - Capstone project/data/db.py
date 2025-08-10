import os
import sqlite3
import logging
from typing import List, Dict, Any

# Путь к БД — оставляем как в проекте
DB_PATH = "retail.db"

# Централизованный логгер БД
logger = logging.getLogger("db")

def _conn():
    """Возвращает соединение с БД с row_factory=Row."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db() -> None:
    """Создаёт таблицу orders и один раз наполняет демо-данными."""
    need_seed = not os.path.exists(DB_PATH)
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_date TEXT,
                customer   TEXT,
                city       TEXT,
                product    TEXT,
                quantity   INTEGER,
                price      REAL
            );
            """
        )
        if need_seed:
            sample = [
                ("2025-07-01", "Иван Петров",       "Moscow",       "Ноутбук",    1, 1200.00),
                ("2025-07-01", "ООО Ромашка",       "Saint Petersburg","Монитор",  3,  220.00),
                ("2025-07-02", "Анна Смирнова",     "Kazan",        "Клавиатура", 2,   45.90),
                ("2025-07-03", "Павел Орлов",       "Moscow",       "Мышь",       1,   25.00),
                ("2025-07-03", "Tech LLC",          "Novosibirsk",  "Сервер",     1, 3900.00),
                ("2025-07-04", "ИП Сидоров",        "Moscow",       "Монитор",    5,  210.00),
                ("2025-07-04", "Мария Соколова",    "Kazan",        "Ноутбук",    1, 1150.00),
            ]
            cur.executemany(
                "INSERT INTO orders(order_date, customer, city, product, quantity, price) VALUES(?,?,?,?,?,?)",
                sample,
            )
        conn.commit()

def safe_select(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """
    Выполняет ТОЛЬКО SELECT, логирует запросы и возвращает список dict-строк.
    Бросает исключение при небезопасных конструкциях.
    """
    sql_norm = sql.strip().lower()
    if not sql_norm.startswith("select"):
        raise ValueError("Разрешены только SELECT запросы.")
    # простая защита
    bad_markers = [" drop ", " delete ", " update ", " insert ", " alter ", ";--"]
    if any(m in f" {sql_norm} " for m in bad_markers):
        raise ValueError("Найдено небезопасное слово в SQL.")

    logger.info("SQL> %s | params=%s", sql.strip(), params)

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

def kpi_revenue() -> float:
    """Суммарная выручка."""
    row = safe_select("SELECT SUM(quantity*price) AS revenue FROM orders")[0]
    return float(row["revenue"] or 0.0)

def kpi_orders() -> int:
    """Количество заказов."""
    row = safe_select("SELECT COUNT(*) AS cnt FROM orders")[0]
    return int(row["cnt"])

def top_products(limit: int = 5) -> List[Dict[str, Any]]:
    """ТОП товаров по выручке."""
    return safe_select(
        """
        SELECT product, SUM(quantity) AS qty, ROUND(SUM(quantity*price),2) AS revenue
        FROM orders
        GROUP BY product
        ORDER BY revenue DESC
        LIMIT ?
        """,
        (limit,),
    )
