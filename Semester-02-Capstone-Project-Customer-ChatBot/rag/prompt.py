from __future__ import annotations
from typing import List
from dataclasses import dataclass
from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "Ты — корпоративный ассистент службы поддержки. Отвечай ТОЛЬКО на основе предоставленных фрагментов документов. "
    "Если ответ в фрагментах не найден — скажи об этом честно и предложи создать тикет. "
    "Никаких домыслов, внешних знаний и ссылок на интернет. Обязательно указывай цитаты источников (файл и страница)."
)

@dataclass
class PromptBundle:
    """Пара system/user для LLM."""
    system: str
    user: str

def format_context(chunks: List[RetrievedChunk]) -> str:
    """Сериализует фрагменты в текстовый блок с цитатами и score."""
    lines = []
    for i, c in enumerate(chunks, 1):
        head = f"[#${i} | score={c.score:.2f} | source={c.source}{' p.' + str(c.page) if c.page else ''}]"
        lines.append(head)
        lines.append(c.text)
        lines.append("---")
    return "\n".join(lines)

def build_prompt(user_question: str, chunks: List[RetrievedChunk], company_info: dict) -> PromptBundle:
    """Собирает system/user промпты, включая блок сведений о компании и контекст с цитатами."""
    context = format_context(chunks) if chunks else "(контекст не найден)"
    company_block = (
        f"Компания: {company_info.get('name')} | Email: {company_info.get('contact_email')} | "
        f"Телефон: {company_info.get('contact_phone')} | Сайт: {company_info.get('homepage')}"
    )
    user = (
        "Формат ответа: сначала короткий вывод, затем чёткие шаги/ссылки на фрагменты в виде [file p.X].\n"
        "Если информации недостаточно — явно скажи об этом.\n\n"
        f"{company_block}\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ВОПРОС: {user_question}"
    )
    return PromptBundle(system=SYSTEM_PROMPT, user=user)
