from __future__ import annotations
from typing import List
from dataclasses import dataclass
from .index import FaissIndex, VectorRecord

@dataclass
class RetrievedChunk:
    """Финальный вид фрагмента для промпта/отображения."""
    text: str
    source: str
    page: int | None
    score: float

class HybridRetriever:
    """Отбор по порогу, dedup и ограничение длины контекста."""
    def __init__(self, index: FaissIndex, min_score: float = 0.3) -> None:
        self.index = index
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 6, max_context_chars: int = 12000) -> List[RetrievedChunk]:
        """Ищет релевантные фрагменты и обрезает общий контекст до max_context_chars."""
        raw = self.index.search(query, k=top_k * 2)
        filtered = [(s, r) for s, r in raw if s >= self.min_score]
        filtered.sort(key=lambda x: x[0], reverse=True)
        picked: List[RetrievedChunk] = []
        total = 0
        for score, rec in filtered:
            if len(rec.text) < 20:
                continue
            if total + len(rec.text) > max_context_chars and picked:
                break
            picked.append(RetrievedChunk(text=rec.text, source=rec.source, page=rec.page, score=score))
            total += len(rec.text)
            if len(picked) >= top_k:
                break
        return picked
