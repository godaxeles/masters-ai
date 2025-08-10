from typing import Dict, Any
from data.db import kpi_revenue, kpi_orders, top_products

def get_kpis(top_n: int = 3) -> Dict[str, Any]:
    """
    Возвращает ключевые KPI из БД: выручка, количество заказов и топ N товаров.
    """
    rev = kpi_revenue()
    cnt = kpi_orders()
    top = top_products(top_n)
    return {
        "ok": True,
        "revenue": rev,
        "orders": cnt,
        "top_products": top,
    }
