from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .utils import recursive_split, hash_text
from .loader import DocumentLoader, DocChunk

@dataclass
class VectorRecord:
    """Эмбеддинг + метаданные для одного чанка."""
    id: str
    embedding: np.ndarray | None
    source: str
    page: int | None
    text: str

class FaissIndex:
    """Персистентный FAISS‑индекс на косинусной близости (inner product по нормированным векторам)."""
    def __init__(self, index_dir: str = "indexes/faiss_index",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.records: List[VectorRecord] = []

    def build_from_loader(self, loader: DocumentLoader,
                          chunk_size: int = 900,
                          chunk_overlap: int = 150) -> None:
        """Сканирует data/, режет на чанки, считает эмбеддинги и строит FAISS."""
        files = loader.list_files()
        if not files:
            raise RuntimeError("В каталоге data/ нет поддерживаемых файлов.")
        chunks: List[DocChunk] = []
        for path in files:
            for page_chunk in loader.load_file(path):
                for piece in recursive_split(page_chunk.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                    chunks.append(DocChunk(text=piece, source=page_chunk.source, page=page_chunk.page))
        texts = [c.text for c in chunks]
        embs = self.model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        self.records = []
        for emb, c in zip(embs, chunks):
            rid = hash_text(c.text + (c.source or "") + str(c.page))
            self.records.append(VectorRecord(id=rid, embedding=emb, source=c.source, page=c.page, text=c.text))
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    def save(self) -> None:
        """Сохраняет FAISS‑индекс и метаданные на диск."""
        if self.index is None:
            raise RuntimeError("Индекс ещё не построен.")
        faiss.write_index(self.index, os.path.join(self.index_dir, "index.faiss"))
        meta = [{"id": r.id, "source": r.source, "page": r.page, "text": r.text} for r in self.records]
        with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        with open(os.path.join(self.index_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f, ensure_ascii=False)

    def load(self) -> None:
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "meta.json")
        info_path = os.path.join(self.index_dir, "info.json")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise RuntimeError("Персистентный индекс не найден. Соберите его (scripts/build_index.py).")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.records = [VectorRecord(id=m["id"], embedding=None, source=m["source"], page=m.get("page"), text=m["text"])
                        for m in meta]
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            model_name = info.get("model_name", self.model_name)
            if model_name != self.model_name:
                self.model_name = model_name
                self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, k: int = 6) -> List[Tuple[float, VectorRecord]]:
        """Возвращает k лучших совпадений (score, record)."""
        if self.index is None:
            raise RuntimeError("Индекс не загружен/не построен.")
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idxs = self.index.search(q, k)
        out: List[Tuple[float, VectorRecord]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            out.append((float(score), rec))
        return out
