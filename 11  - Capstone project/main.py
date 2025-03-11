import streamlit as st
import sqlite3
import openai
from utils.database_utils import get_employee_data
from utils.email_utils import send_email
from utils.ai_utils import get_ai_response
import pandas as pd
import matplotlib.pyplot as plt

# Настройка UI
st.set_page_config(page_title='HR Assistant', layout='wide')
st.title('HR AI Assistant')

# Подключение к базе данных
conn = sqlite3.connect('employees.db')
cursor = conn.cursor()

# Фильтрация данных
st.sidebar.header('Фильтр сотрудников')
departments = [row[0] for row in cursor.execute("SELECT DISTINCT department FROM employees").fetchall()]
selected_dept = st.sidebar.selectbox('Выберите отдел', ['Все'] + departments)

# Загрузка данных
employees = get_employee_data(conn, selected_dept)
st.dataframe(employees)

# Аналитика
st.subheader("Статистика по сотрудникам")
fig, ax = plt.subplots()
department_counts = employees['department'].value_counts()
department_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Генерация AI-ответов
st.subheader("AI Анализ")
question = st.text_input("Введите ваш запрос")
if st.button("Получить ответ"):
    response = get_ai_response(question)
    st.write(response)

# Отправка email
st.subheader("Отправка Email")
recipient = st.text_input("Email получателя")
message = st.text_area("Сообщение")
if st.button("Отправить"):
    send_email(recipient, message)
    st.success("Сообщение отправлено!")

conn.close()
