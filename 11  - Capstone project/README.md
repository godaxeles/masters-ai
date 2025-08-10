
# 🤖 Capstone Generative AI Agent (чистая архитектура)

## Установка
```bash
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
# cmd: .venv\Scripts\activate.bat
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка
Создайте `.env` в корне проекта:
```
OPENAI_API_KEY=sk-...    # без кавычек
OPENAI_MODEL=gpt-4o-mini
```

## Активация окружения
```bash
.\.venv\Scripts\Activate.ps1
```

## Запуск
```bash
streamlit run app.py
```

## Архитектура
- `config/` — загрузка настроек (.env / secrets)
- `core/` — логирование
- `data/` — доступ к БД и KPI
- `tools/` — инструменты (SQL и погода)
- `llm/client.py` — инициализация OpenAI (без прокси, совместимо с httpx 0.27/0.28+)
- `llm/agent.py` — function calling и orchestration
- `app.py` — только UI Streamlit
