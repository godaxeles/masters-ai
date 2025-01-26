import streamlit as st
from utils import execute_query, parse_user_query, trigger_external_api
import logging

# Настройка логирования
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(message)s")

st.title("Capstone Project: Умный агент")
st.sidebar.header("Чат-агент")

st.sidebar.subheader("Ваш запрос")
user_input = st.sidebar.text_area("Введите запрос")

if st.sidebar.button("Отправить"):
    query, params = parse_user_query(user_input)

    if query == "API_CALL":
        response = trigger_external_api({"query": user_input})
        st.write("Ответ API:")
        st.write(response)

    elif query:
        logging.info("SQL-запрос: %s, параметры: %s", query, params)
        result = execute_query(query, params)
        if not result.empty:
            st.write("Результат запроса:")
            st.write(result)
        else:
            st.write("Нет данных по вашему запросу.")

    else:
        st.write("Запрос не поддерживается. Попробуйте другой запрос.")
