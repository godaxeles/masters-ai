import os
import openai
import requests
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from conversation import Conversation

# Загрузка переменных окружения
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
DATABASE = "products_data.db"

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def ask_database(conn, query):
    """Функция для выполнения SQL-запроса в базе данных SQLite."""
    try:
        results = conn.execute(query).fetchall()
        return results
    except Exception as e:
        raise Exception(f"SQL error: {e}")


def plot_price_trend(product_code, conn):
    """Функция для построения графика изменения цены продукта."""
    # SQL запрос для получения данных
    query = f"""
    SELECT month, price
    FROM products
    WHERE product_code = '{product_code}'
    ORDER BY month;
    """
    # Выполнение запроса
    data = pd.read_sql(query, conn)
    
    # Проверяем, что данные получены
    if data.empty:
        return None

    # Попробуем преобразовать месяц в datetime, учитывая возможный формат '01-2024'
    data['month'] = pd.to_datetime(data['month'], format='%m-%Y', errors='coerce')  # формат 'MM-YYYY'

    # Проверка, есть ли проблемы с преобразованием
    if data['month'].isnull().any():
        st.error("There was an issue with parsing the date. Please check the format of the 'month' field.")
        return None

    # Создаем график
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='price', data=data, marker='o')

    plt.title(f"Price Trend for Product Code {product_code}")
    plt.xlabel('Month')
    plt.ylabel('Price ($)')
    plt.grid(True)

    # Отображаем график в Streamlit
    st.pyplot(plt)


def chat_completion_with_function_execution(messages, functions=None):
    """Функция для вызова ChatCompletion API и выполнения функции, если она необходима."""
    try:
        response = chat_completion_request(messages, functions)
        full_message = response.json()["choices"][0]
        if full_message["finish_reason"] == "function_call":
            print(f"Function generation requested, calling function")
            return call_function(messages, full_message)
        else:
            print(f"Function not required, responding to user")
            return response.json()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return {}


def call_function(messages, full_message):
    """Выполняет вызов функции, если требуется."""
    if full_message["message"]["function_call"]["name"] == "ask_database":
        query = eval(full_message["message"]["function_call"]["arguments"])
        print(f"Prepped query is {query}")
        try:
            results = ask_database(conn, query["query"])
        except Exception as e:
            print(e)
            messages.append(
                {
                    "role": "system",
                    "content": f"""Query: {query['query']}
The previous query received the error {e}. 
Please return a fixed SQL query in plain text.
Your response should consist of ONLY the SQL query with the separator sql_start at the beginning and sql_end at the end""",
                }
            )
            response = chat_completion_request(messages, model=MODEL)

            try:
                cleaned_query = response.json()["choices"][0]["message"][
                    "content"
                ].split("sql_start")[1]
                cleaned_query = cleaned_query.split("sql_end")[0]
                results = ask_database(conn, cleaned_query)
                print("Got on second try")
            except Exception as e:
                print("Second failure, exiting")
                print(f"Function execution failed")
                print(f"Error message: {e}")

        messages.append(
            {"role": "function", "name": "ask_database", "content": str(results)}
        )

        try:
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            print(e)
            raise Exception("Function does not exist and cannot be called")


# Системное сообщение для базы данных
agent_system_message = """You are DatabaseGPT, a helpful assistant who gets answers to user questions from the Database.
Provide as many details as possible to your users.
Begin!"""

# Основная часть
if __name__ == '__main__':
    conversation = Conversation()
    conn = sqlite3.connect(DATABASE)

    # Схема базы данных
    database_schema_string = """
    Table: products
    Columns: id, product_code, product_name, category, color, capacity, price, monthly_sales, stock_remaining, segment, discount, country_of_origin, product_index, flag, month
    """

    # Функция запроса
    functions = [
        {
            "name": "ask_database",
            "description": "Use this function to answer user questions about data. Output should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    ]

    # Добавляем системное сообщение
    conversation.add_message("system", agent_system_message)

    # Получаем ввод от пользователя через Streamlit (или консоль)
    st.title("Database Query Assistant")

    # История переписки
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Инициализация ключа query_input
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Определение фонов для разных ролей
    role_backgrounds = {
        "user": "#DCF8C6",  # светло-зеленый фон для пользователя
        "assistant": "#F1F1F1",  # светло-серый фон для ассистента
        "function": "#F5F5F5",  # светло-серый фон для функции
        "system": "#E6E6FA"  # лавандовый фон для системы
    }

    # Отображаем историю сообщений
    for message in st.session_state["messages"]:
        background_color = role_backgrounds.get(message['role'], '#FFFFFF')  # По умолчанию белый фон
        alignment = 'right' if message['role'] == "user" else 'left'  # Вопросы пользователя справа, ответы слева

        st.markdown(f"<div style='background-color:{background_color}; padding: 12px; border-radius: 8px; box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 4px; text-align:{alignment};'>"
                    f"<b style='color: black;'>{message['role'].capitalize()}:</b><br><span style='color: black;'>{message['content']}</span></div>", unsafe_allow_html=True)

    # Создание колонок для поля ввода и кнопки
    col1, col2 = st.columns([4, 1])

    # Поле для ввода запроса
    with col1:
        query_input = st.text_input("", key="query_input", label_visibility="collapsed", placeholder="Enter your query here...")

    # Кнопка для отправки запроса
    with col2:
        submit_button = st.button("Send Query")

    # Добавление нового сообщения при вводе и нажатии кнопки
    if submit_button and query_input:
        # Добавляем флаг, чтобы предотвратить повторное выполнение запроса
        if query_input not in [msg['content'] for msg in st.session_state["messages"]]:
            conversation.add_message("user", query_input)
            st.session_state["messages"].append({"role": "user", "content": query_input})

            chat_response = chat_completion_with_function_execution(
                conversation.conversation_history, functions=functions
            )

            try:
                assistant_message = chat_response["choices"][0]["message"]["content"]
                conversation.add_message("assistant", assistant_message)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_message})

                # Переключаем историю сообщений вниз
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")   
