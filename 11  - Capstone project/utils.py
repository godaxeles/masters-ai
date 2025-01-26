import sqlite3
import pandas as pd
import requests


# Функция для выполнения SQL-запросов
def execute_query(query, params=()):
    connection = sqlite3.connect("data/data.db")
    try:
        data = pd.read_sql(query, connection, params=params)
    except Exception as e:
        data = pd.DataFrame({"Error": [str(e)]})
    connection.close()
    return data


# Логика разбора пользовательского запроса
def parse_user_query(user_input):
    user_input = user_input.lower()

    if "все компании" in user_input:
        return "SELECT * FROM business_data", ()

    elif "прибыль выше" in user_input:
        try:
            min_profit = int("".join(filter(str.isdigit, user_input)))
            return "SELECT name, profit FROM business_data WHERE profit > ?", (min_profit,)
        except ValueError:
            return None, None

    elif "доход выше" in user_input:
        try:
            min_revenue = int("".join(filter(str.isdigit, user_input)))
            return "SELECT name, revenue FROM business_data WHERE revenue > ?", (min_revenue,)
        except ValueError:
            return None, None

    elif "самая прибыльная компания" in user_input:
        return "SELECT name, MAX(profit) AS max_profit FROM business_data", ()

    elif "наименьший доход" in user_input:
        return "SELECT name, MIN(revenue) AS min_revenue FROM business_data", ()

    elif "средняя прибыль" in user_input:
        return "SELECT AVG(profit) AS avg_profit FROM business_data", ()

    elif "отсортируй по прибыли" in user_input:
        return "SELECT name, revenue, profit FROM business_data ORDER BY profit DESC", ()

    elif "отсортируй по доходу" in user_input:
        return "SELECT name, revenue, profit FROM business_data ORDER BY revenue DESC", ()

    elif "вывод только названия" in user_input:
        return "SELECT name FROM business_data", ()

    elif "вызов api" in user_input:
        return "API_CALL", None  # Для вызова API через другую функцию

    else:
        return None, None


# Функция для вызова внешнего API
def trigger_external_api(data):
    try:
        # Замените URL на реальный API, если потребуется
        url = "https://jsonplaceholder.typicode.com/posts"
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
