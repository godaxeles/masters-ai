from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional
from pdfminer.high_level import extract_text as pdf_extract_text
import markdown as md
from .utils import normalize_ws

@dataclass
class DocChunk:
    """Фрагмент исходного документа (для индексации)."""
    text: str
    source: str
    page: Optional[int] = None

class DocumentLoader:
    """Загружает PDF/TXT/MD из каталога data/ и превращает их в чистый текст."""
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir

    def list_files(self) -> List[str]:
        """Возвращает список путей ко всем поддерживаемым файлам."""
        supported_ext = {".pdf", ".txt", ".md"}
        found: List[str] = []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_ext:
                    found.append(os.path.join(root, f))
        return sorted(found)

    def load_file(self, path: str) -> List[DocChunk]:
        """Загружает один файл и возвращает список DocChunk с нормализованным текстом."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = normalize_ws(f.read())
            return [DocChunk(text=text, source=os.path.basename(path), page=None)]
        elif ext == ".md":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                text = md.markdown(raw)
            text = normalize_ws(_strip_html_tags(text))
            return [DocChunk(text=text, source=os.path.basename(path), page=None)]
        else:
            return []

    def _load_pdf(self, path: str) -> List[DocChunk]:
        """Извлекает текст из PDF. Для цитат пытаемся оценить страницы (по \\f)."""
        text = pdf_extract_text(path) or ""
        pages = text.split("\f") if "\f" in text else [text]
        chunks: List[DocChunk] = []
        for i, p in enumerate(pages, start=1):
            chunks.append(DocChunk(text=normalize_ws(p),
                                   source=os.path.basename(path),
                                   page=i))
        return chunks

def _strip_html_tags(html: str) -> str:
    """Грубое удаление HTML‑тегов (после markdown → html)."""
    import re
    return re.sub(r"<[^>]+>", " ", html)
