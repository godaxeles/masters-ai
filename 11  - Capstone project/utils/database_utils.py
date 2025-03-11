import sqlite3
import pandas as pd

DB_PATH = 'employees.db'

def connect_db():
    """Устанавливает соединение с базой данных и создает таблицу, если ее нет."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department TEXT
        )
    """)
    conn.commit()
    return conn

def get_employee_data(conn, department=None):
    """Получает данные о сотрудниках."""
    query = "SELECT * FROM employees"
    params = ()
    if department and department != 'Все':
        query += " WHERE department = ?"
        params = (department,)
    
    df = pd.read_sql_query(query, conn, params=params)
    return df
