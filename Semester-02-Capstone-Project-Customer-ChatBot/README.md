---
title: Customer Support RAG
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---


# RAG Customer Support Bot — «только по данным» (Streamlit + FAISS + цитаты)

Полноценный проект RAG‑бота, который **отвечает исключительно по вашим документам**.
- Индексация PDF/TXT/MD (`pdfminer.six`), векторный поиск FAISS + эмбеддинги `all-MiniLM-L6-v2`
- Жёсткое принуждение «только по данным»: системный промпт + пост‑проверка наличия цитат
- UI на Streamlit с историей диалога и источниками/страницами
- Создание тикетов (Jira/Trello/GitHub) или локальный fallback (`tickets.jsonl`)

## 1) Требования
- **Python 3.11** (Windows: рекомендуется 3.11 для совместимости с faiss-cpu)
- PowerShell (Windows) или bash (Linux/Mac)

## 2) Установка (Windows PowerShell)
```powershell
# из корня проекта
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## 3) Подготовьте данные
Положите ваши документы в папку `data/` (поддерживается .pdf / .txt / .md).
В архиве есть пример `data/sample.md` — можно начать с него, затем замените своими файлами.

## 4) Соберите индекс
```powershell
python .\scripts\build_index.py --rebuild
```
После сборки появятся файлы:
```
indexes/faiss_index/index.faiss
indexes/faiss_index/meta.json
```

## 5) Настройте LLM‑провайдера (любой один из вариантов)
### Вариант A. OpenAI (например, gpt-4o-mini)
```powershell
$env:OPENAI_API_KEY = "sk-ВашКлюч"
$env:LLM_PROVIDER   = "openai"
$env:LLM_MODEL      = "gpt-4o-mini"
```
### Вариант B. Hugging Face Inference (например, Llama‑3‑8B‑Instruct)
```powershell
$env:HF_API_TOKEN = "hf_ВашТокен"
$env:LLM_PROVIDER = "hf"
$env:LLM_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct"
```

> Если ключи не заданы, приложение запустится и подскажет, какие переменные нужно установить.

## 6) Запуск UI
```powershell
streamlit run .\app.py
```

Откроется браузер `http://localhost:8501`. Слева — настройки, в центре — чат.
Если ответа «по данным» нет — бот честно скажет об этом и предложит создать тикет.

## 7) Создание тикетов (опционально)
Задайте переменные, если хотите отправлять в внешние системы (иначе запись в `tickets/tickets.jsonl`):
```powershell
# Jira
setx JIRA_BASE_URL "https://your.atlassian.net"
setx JIRA_EMAIL "you@example.com"
setx JIRA_API_TOKEN "token"
setx JIRA_PROJECT_KEY "SUP"

# Trello
setx TRELLO_KEY "key"
setx TRELLO_TOKEN "token"
setx TRELLO_LIST_ID "list_id"

# GitHub Issues
setx GITHUB_TOKEN "ghp_..."
setx GITHUB_REPO "owner/repo"
```

## 8) Диагностика
- Нет `index.faiss` → запустите сборку индекса (шаг 4) и убедитесь, что в `data/` есть файлы.
- Ошибка установки `faiss-cpu`/`numpy` → проверьте, что активирован Python **3.11** в `.venv`.
- LLM‑ошибки авторизации → задайте переменные окружения из шага 5 и перезапустите Streamlit.
