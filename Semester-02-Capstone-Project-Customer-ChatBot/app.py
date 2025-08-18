from __future__ import annotations
import os, yaml
import streamlit as st
from typing import List
from rag.index import FaissIndex
from rag.retriever import HybridRetriever
from rag.prompt import build_prompt
from rag.llm import LLM, LLMConfig
from rag.ticketing import Ticketing, Ticket
from dotenv import load_dotenv, find_dotenv
IN_HF_SPACE = bool(os.environ.get("SPACE_ID", ""))   # признак, что код крутится в Space
load_dotenv(find_dotenv(), override=not IN_HF_SPACE)


@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    """Единоразово загружает config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=True)
def load_index() -> FaissIndex:
    """
    Пытаемся загрузить готовый индекс. Если его нет и мы в Hugging Face Space,
    автоматически собираем из ./data и сохраняем.
    """
    idx = FaissIndex()
    try:
        idx.load()
        return idx
    except Exception as e:
        # если индекс отсутствует — собираем в рантайме
        IN_HF_SPACE = bool(os.environ.get("SPACE_ID", ""))
        if IN_HF_SPACE:
            from rag.loader import DocumentLoader
            st.info("Персистентный индекс не найден. Собираю из ./data прямо в Space…")
            loader = DocumentLoader("data")
            idx.build_from_loader(loader, chunk_size=900, chunk_overlap=150)
            idx.save()
            st.success("Индекс собран и сохранён.")
            return idx
        # локально продолжаем бросать исключение (как было)
        raise



@st.cache_resource(show_spinner=False)
def init_llm(cfg: dict) -> LLM | None:
    """Инициализирует LLM. Если ключей нет — возвращает None, а UI покажет подсказку."""
    provider = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    if not provider:
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else cfg.get("provider_fallback", "hf")
    if not model:
        model = "gpt-4o-mini" if provider == "openai" else "meta-llama/Meta-Llama-3-8B-Instruct"
    try:
        llm_conf = LLMConfig(provider=provider, model=model,
                             temperature=cfg.get("temperature", 0.1),
                             max_tokens=cfg.get("max_tokens", 700))
        return LLM(llm_conf)
    except Exception as e:
        st.warning(f"LLM не инициализирован: {e}. Установите переменные окружения для OpenAI или HF (см. README).")
        return None

def user_asked_ticket(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["создай тикет", "создать тикет", "тикет", "ticket", "issue", "jira", "trello", "github"]
    return any(k in t for k in triggers)

def answer_with_flag(llm, prompt, chunks):
    """
    Возвращает (answer_text, not_found: bool).
    Если контекста нет или он пустой — not_found=True и отдаём честный ответ без источников.
    """
    if not chunks:  # ничего не нашлось для контекста
        return ("Я не нашёл достаточной информации в предоставленных документах. "
                "Могу оформить тикет, чтобы специалист вернулся с точным ответом."), True
    # Контекст есть — спрашиваем LLM
    ans = llm.chat(prompt.system, prompt.user)
    return ans, False


# ------------------------- UI -------------------------
st.set_page_config(page_title="Contoso Motors — Support RAG", page_icon="🛠️", layout="wide")
config = load_config()
company = config.get("company", {})
retrieval_cfg = config.get("retrieval", {})
llm_cfg = config.get("llm", {})
ui_cfg = config.get("ui", {})

st.title("🛠️ Customer Support RAG — только по данным")
st.caption(f"Компания: {company.get('name')} | Email: {company.get('contact_email')} | Тел.: {company.get('contact_phone')}")

with st.sidebar:
    st.header("Настройки")
    st.write("LLM и индекс")
    top_k = st.slider("Top‑K документов", min_value=2, max_value=12, value=int(retrieval_cfg.get("top_k", 6)))
    min_score = st.slider("Порог совпадения", 0.0, 1.0, float(retrieval_cfg.get("min_score", 0.3)), 0.01)
    max_ctx = st.slider("Макс. размер контекста", 2000, 20000, int(retrieval_cfg.get("max_context_chars", 12000)), 500)
    temperature = st.slider("Температура", 0.0, 1.0, float(llm_cfg.get("temperature", 0.1)), 0.05)
    if st.button("Пересобрать индекс из data/", type="secondary"):
        st.info("Запустите локально: `python scripts/build_index.py --rebuild` (в Space недоступно)")

# Кэшируем/грузим индекс и LLM
index = load_index()
retriever = HybridRetriever(index, min_score=min_score)
llm = init_llm({"provider_fallback": llm_cfg.get("provider_fallback", "hf"),
                "temperature": temperature,
                "max_tokens": llm_cfg.get("max_tokens", 700)})

if "history" not in st.session_state:
    st.session_state.history = []  # [(user, assistant)]

# Предупреждение, если нет LLM
if llm is None:
    st.warning("LLM не настроен. Установите OPENAI_API_KEY или HF_API_TOKEN и переменные LLM_PROVIDER/LLM_MODEL (см. README).")

# Ввод по Enter
q = st.chat_input("Спросите о документах компании…")  # Enter отправляет
clear = st.button("Очистить историю", type="secondary")

if clear:
    st.session_state.history = []
    st.experimental_rerun()

# История
for i, (u, a) in enumerate(st.session_state.history[-8:]):
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(a)

if q and llm is not None:
    with st.spinner("Ищу ответ по вашим документам…"):
        chunks = retriever.retrieve(q, top_k=top_k, max_context_chars=max_ctx)
        prompt = build_prompt(q, chunks, company)
        answer, not_found = answer_with_flag(llm, prompt, chunks)

    st.chat_message("user").write(q)
    st.chat_message("assistant").write(answer)

    # Источники показываем ТОЛЬКО если ответ найден
    if not not_found and chunks:
        with st.expander("Показать использованные фрагменты (контекст)"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**#{i} — {c.source}{(' p.' + str(c.page)) if c.page else ''} | score={c.score:.2f}**")
                st.write(c.text)
                st.divider()

    # Сохраняем флаг, чтобы решать — показывать форму тикета или нет
    st.session_state.history.append((q, answer))
    st.session_state.last_not_found = not_found

# Очистка истории
if clear:
    st.session_state.history = []
    st.session_state.last_not_found = False
    st.experimental_rerun()


# Блок тикетов
# Показываем форму тикета только если пользователь сам попросил
# ИЛИ если предыдущий ответ был not_found
show_ticket_form = False
if "last_not_found" in st.session_state and st.session_state.last_not_found:
    show_ticket_form = True
if q and user_asked_ticket(q):
    show_ticket_form = True

if show_ticket_form:
    st.subheader("Создать тикет")
    with st.form("ticket_form"):
        name = st.text_input("Ваше имя")
        email = st.text_input("Ваш email")
        title = st.text_input("Заголовок")
        desc = st.text_area("Описание проблемы")
        submit = st.form_submit_button("Отправить")
    if submit:
        if not (name and email and title and desc):
            st.error("Заполните все поля, пожалуйста.")
        else:
            tk = Ticketing()
            ident = tk.create(Ticket(name=name, email=email, title=title, description=desc))
            st.success(f"Создан тикет: {ident}")
