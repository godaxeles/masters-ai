import os
import sqlite3
import streamlit as st
import pandas as pd
from openai import OpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from dotenv import load_dotenv
import requests
import re

# Загрузка переменных окружения
load_dotenv()
st.set_page_config(layout="wide")

# Константы
DATABASE_PATH = os.getenv("DATABASE_PATH", "buildings_with_coordinates.db")
OPENAI_KEY = os.getenv("OPENAI_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CURRENCY_CODES = ["USD", "EUR", "JPY", "RUB", "UZS"]

# Инициализация OpenAI
client = OpenAI(api_key=OPENAI_KEY)

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def fetch_rate(base: str, target: str) -> float:
    resp = requests.get(
        "https://api.freecurrencyapi.com/v1/latest",
        params={"apikey": CURRENCY_API_KEY, "base_currency": base, "currencies": target}
    )
    resp.raise_for_status()
    return resp.json().get("data", {}).get(target, 1.0)


def run_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(query, conn)


def ask_openai(prompt: str) -> str:
    cache = st.session_state.setdefault('query_cache', {})
    if prompt in cache:
        return cache[prompt]
    cols = ['project_no','city','building_name','price_per_sqm','latitude','longitude']
    system_msg = (
        f"You are an assistant that generates only correct SQL queries for the table 'buildings_with_coordinates'."
        f" Columns: {', '.join(cols)}."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            max_tokens=150, temperature=0
        )
    except OpenAIError as e:
        err = str(e)
        if 'insufficient_quota' in err or 'quota' in err:
            raise RuntimeError("Недостаточно квоты API OpenAI.")
        if 'model' in err and 'not exist' in err:
            resp = client.chat.completions.create(
                model='gpt-3.5-turbo', messages=messages,
                max_tokens=150, temperature=0
            )
        else:
            raise
    raw = resp.choices[0].message.content.strip()
    clean = re.sub(r"(?i)^```sql|```$", "", raw).lstrip("`'\" ")
    if clean.lower().startswith("sql"):
        clean = clean[3:].lstrip(" ;").strip()
    clean = re.sub(r"\bFROM\s+buildings\b", "FROM buildings_with_coordinates", clean, flags=re.IGNORECASE)
    clean = re.sub(r"(?i)SELECT\s+.*?FROM", "SELECT * FROM", clean, count=1)
    cache[prompt] = clean
    return clean


def main():
    st.title("Поиск недвижимости и конвертация валют")
    base_cur = st.sidebar.selectbox("Базовая валюта", CURRENCY_CODES)
    target_cur = st.sidebar.selectbox("Целевая валюта", CURRENCY_CODES, index=4)
    # Инициализация
    if 'df' not in st.session_state:
        st.session_state.df = run_sql("SELECT * FROM buildings_with_coordinates;")
    # Форма запроса
    st.sidebar.subheader("Задать вопрос")
    question = st.sidebar.text_area("Что вы хотите найти?", height=100)
    if st.sidebar.button("Выполнить запрос"):
        try:
            sql = ask_openai(question)
            df = run_sql(sql)
                        # Преобразование только числовых колонок
            numeric_fields = ['price_per_sqm','min_price_per_sqm','max_price_per_sqm','latitude','longitude']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            # Конвертация валют
            try:
                rate = fetch_rate(base_cur, target_cur)
            except RetryError:
                rate = 1.0
                st.warning("Не удалось получить курс валют, используется 1.0")
            if 'price_per_sqm' in df.columns:
                df['price_converted'] = df['price_per_sqm'] * rate
            st.session_state.df = df
        except Exception as e:
            st.error(str(e))
    # Основная таблица и карта
    st.subheader("Таблица результатов")
    df_disp = st.session_state.df.copy().astype(str)
    st.dataframe(df_disp)
    if {'latitude','longitude'}.issubset(st.session_state.df.columns):
        coords = st.session_state.df.copy()
        coords['latitude']=pd.to_numeric(coords['latitude'],errors='coerce')
        coords['longitude']=pd.to_numeric(coords['longitude'],errors='coerce')
        coords=coords.dropna(subset=['latitude','longitude'])
        if not coords.empty:
            st.map(coords[['latitude','longitude']])

if __name__ == '__main__':
    main()
