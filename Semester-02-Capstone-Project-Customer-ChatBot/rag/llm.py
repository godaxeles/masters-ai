from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from huggingface_hub import InferenceClient

@dataclass
class LLMConfig:
    """Конфигурация LLM‑провайдера."""
    provider: str  # "openai" или "hf"
    model: str
    temperature: float = 0.1
    max_tokens: int = 700

class LLM:
    """Мини‑абстракция над OpenAI и HF Inference API."""
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._openai: Optional[OpenAI] = None
        self._hf: Optional[InferenceClient] = None

        if cfg.provider == "openai":
            self._openai = OpenAI()
        elif cfg.provider == "hf":
            token = os.environ.get("HF_API_TOKEN")
            if not token:
                raise RuntimeError("Для провайдера hf требуется переменная окружения HF_API_TOKEN.")
            self._hf = InferenceClient(token=token)
        else:
            raise ValueError("Неизвестный провайдер LLM. Используйте 'openai' или 'hf'.")

    def chat(self, system: str, user: str) -> str:
        """Выполняет диалоговый запрос и возвращает текст ответа модели."""
        if self.cfg.provider == "openai":
            assert self._openai is not None
            resp = self._openai.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        else:
            # HF Inference: text-generation с chat‑форматом
            assert self._hf is not None
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>"
            resp = self._hf.text_generation(
                prompt,
                model=self.cfg.model,
                max_new_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                do_sample=True,
                return_full_text=False,
            )
            return resp
