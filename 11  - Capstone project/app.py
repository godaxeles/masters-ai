import re
import logging
import streamlit as st

from data.db import init_db, kpi_revenue, kpi_orders, top_products
from llm.agent import chat, SYSTEM_PROMPT
from llm.client import get_openai_client

# ----------------- Настройки и логирование -----------------
st.set_page_config(page_title="Capstone Generative AI Agent", page_icon="🤖", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("app")

# ----------------- Декор (CSS) -----------------
st.markdown(
    """
<style>
:root{
  --bg:#0f1117; --panel:#151a23; --card:#11151d; --muted:#a3a7b7;
  --text:#f1f5f9; --accent:#7c3aed; --accent2:#06b6d4; --good:#22c55e;
}
html, body, .stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

h1.title {
  font-size: 2rem; font-weight: 800; letter-spacing: .2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: .2rem 0 1rem 0;
}
.badge {
  display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .6rem; border-radius:999px;
  background:#1b2130; border:1px solid #232c40; font-size:.85rem; color:var(--muted);
}
.badge .dot{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.badge .on{ background: var(--good); box-shadow:0 0 10px var(--good); }
.badge .off{ background: #ef4444; box-shadow:0 0 10px #ef4444; }

.section { background: #121827; border: 1px solid #232c40; border-radius: 14px; padding: .8rem; }

.kpi-card {
  background: #11151d; border: 1px solid #21293a; border-radius: 14px; padding: .8rem 1rem; margin-bottom:.6rem;
}
.kpi-title { font-size:.82rem; color:var(--muted); margin:0 0 .15rem 0; }
.kpi-value { font-size:1.2rem; font-weight:700; }

.quickbar { display:flex; gap:.5rem; }
button[kind="secondary"] { border-radius:999px !important; }

.stChatMessage { margin-bottom:.65rem; }
.stChatMessage div[data-testid="stMarkdownContainer"] p { margin:.2rem 0; }
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- Доменные гардрейлы -----------------
REFUSAL_TEXT = "Извините, я могу отвечать только на вопросы по заказам, KPI и погоде."
KWD_RE = re.compile(
    r"(заказ|продаж|выручк|^kpi$|kpi|кейпиай|метрик|показател"
    r"|погод[ауеы]|доставк|логист|топ|товар)",
    re.IGNORECASE
)

def is_allowed_prompt(text: str) -> bool:
    return bool(KWD_RE.search(text or ""))

def guard_answer(text: str) -> str:
    return text if is_allowed_prompt(text or "") else REFUSAL_TEXT

# ----------------- Инициализация -----------------
init_db()

if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]

# гарантированное состояние для быстрых запросов
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = ""

# ----------------- Верхняя панель -----------------
client, model = get_openai_client()
status_html = f"""
<div class="badge">
  <span class="dot {'on' if client else 'off'}"></span>
  <span>{'Online' if client else 'Offline'}</span>
  <span style="opacity:.7">·</span>
  <span>model: <b>{model}</b></span>
</div>
"""
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown('<h1 class="title">🤖 Capstone Generative AI Agent</h1>', unsafe_allow_html=True)
with right:
    st.markdown(status_html, unsafe_allow_html=True)

# ----------------- Быстрые подсказки -----------------
with st.container():
    st.markdown('<div class="section">**Быстрый запрос:**</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    def queue_and_rerun(text: str):
        st.session_state.queued_prompt = text
        st.rerun()

    if c1.button("Покажи все заказы", use_container_width=True):
        queue_and_rerun("Покажи все заказы")
    if c2.button("Дай заказы Ивана", use_container_width=True):
        queue_and_rerun("Дай заказы Ивана")
    if c3.button("Погода для Moscow на сегодня", use_container_width=True):
        queue_and_rerun("Погода для Moscow на сегодня")
    if c4.button("ТОП товары", use_container_width=True):
        queue_and_rerun("ТОП товары")
    if c5.button("Какой KPI?", use_container_width=True):
        queue_and_rerun("Какой KPI?")

# ----------------- Рендер истории (без пустых сообщений) -----------------
for m in st.session_state.history[1:]:
    content = (m.get("content") or "").strip()
    if not content:
        continue
    with st.chat_message(m["role"]):
        st.markdown(content)

# ----------------- Ввод пользователя -----------------
typed = st.chat_input("Спросите про заказы, KPI или погоду для доставки…")
prompt = typed or st.session_state.queued_prompt
# сразу очищаем очередь, чтобы не повторялось
if st.session_state.queued_prompt:
    st.session_state.queued_prompt = ""

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not is_allowed_prompt(prompt):
        text = REFUSAL_TEXT
        st.session_state.history.append({"role": "assistant", "content": text})
        with st.chat_message("assistant"):
            st.markdown(text)
    else:
        result = chat(st.session_state.history)
        if result["type"] == "llm":
            m = result["message"]
            safe_text = (guard_answer(m.content or "")).strip()
            if safe_text:
                st.session_state.history.append({"role": "assistant", "content": safe_text})
                with st.chat_message("assistant"):
                    st.markdown(safe_text)
        else:
            text = (result.get("content", "")).strip()
            if text:
                st.session_state.history.append({"role": "assistant", "content": text})
                with st.chat_message("assistant"):
                    st.markdown(text)

# ----------------- Сайдбар KPI -----------------
with st.sidebar:
    st.markdown("### 📊 Бизнес-панель")
    st.markdown('<div class="kpi-card"><div class="kpi-title">Выручка</div>'
                f'<div class="kpi-value">${kpi_revenue():,.2f}</div></div>', unsafe_allow_html=True)
    cA, cB = st.columns(2)
    with cA:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Количество заказов</div>'
                    f'<div class="kpi-value">{kpi_orders()}</div></div>', unsafe_allow_html=True)
    with cB:
        st.markdown('<div class="kpi-card"><div class="kpi-title">ТОП-товары</div>', unsafe_allow_html=True)
        for r in top_products(5):
            st.markdown(f"- {r['product']} — ${r['revenue']}")
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("Логи пишутся в stdout. Ключи берутся из .env/ENV.")
