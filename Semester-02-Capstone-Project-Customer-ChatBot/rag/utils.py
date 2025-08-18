from __future__ import annotations
import hashlib
import re
from typing import List, Tuple

def normalize_ws(text: str) -> str:
    """Нормализует пробелы/переводы строк для стабильного чанкинга и поиска."""
    return re.sub(r"\s+", " ", text).strip()

def hash_text(text: str) -> str:
    """Возвращает SHA1‑хэш — используется как стабильный id чанка."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def recursive_split(text: str, *, chunk_size: int = 900, chunk_overlap: int = 150,
                    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> List[str]:
    """Рекурсивно делит текст на перекрывающиеся чанки, стараясь не рвать по словам."""
    if len(text) <= chunk_size:
        return [text]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        chunks: List[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                tail = buf[-chunk_overlap:] if buf else ""
                buf = (tail + p)[:chunk_size]
                if len(buf) > chunk_size:
                    for i in range(0, len(buf), chunk_size - chunk_overlap):
                        chunks.append(buf[i:i + chunk_size])
                    buf = ""
        if buf:
            chunks.append(buf)
        out: List[str] = []
        for c in chunks:
            if len(c) <= chunk_size:
                out.append(c)
            else:
                out.extend(recursive_split(c, chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap,
                                           separators=separators[1:]))
        return out
    # Fallback: жёсткая нарезка по символам (с перекрытием)
    res: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(text), step):
        res.append(text[i:i + chunk_size])
    return res
