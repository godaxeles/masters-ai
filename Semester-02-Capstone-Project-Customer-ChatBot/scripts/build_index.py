from __future__ import annotations
import argparse, os, json
from rag.loader import DocumentLoader
from rag.index import FaissIndex

def main() -> None:
    """CLI: сборка/пересборка индекса из каталога data/."""
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default="indexes/faiss_index")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    if args.rebuild:
        import shutil
        if os.path.exists(args.index_dir):
            shutil.rmtree(args.index_dir)

    loader = DocumentLoader("data")
    index = FaissIndex(index_dir=args.index_dir, model_name=args.model)
    index.build_from_loader(loader, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    index.save()
    print("✅ Индекс собран и сохранён в", args.index_dir)

if __name__ == "__main__":
    main()
