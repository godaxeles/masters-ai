"""Vector store service for document embeddings and similarity search."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import structlog

from config import settings
from models.document import DocumentChunk, DocumentMetadata
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS-based vector store for document similarity search."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize vector store.

        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.index = None
        self.metadata = []
        self.dimension = None

        logger.info("Initializing vector store", model=self.model_name)

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info("Loading embedding model", model=self.model_name)
            try:
                self.model = SentenceTransformer(self.model_name)
                # Get model dimension
                test_embedding = self.model.encode(["test"])
                self.dimension = test_embedding.shape[1]
                logger.info("Model loaded successfully", dimension=self.dimension)
            except Exception as e:
                logger.error("Failed to load embedding model", error=str(e))
                raise

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return

        self._load_model()

        logger.info("Adding documents to vector store", chunk_count=len(chunks))

        # Extract text for embedding
        texts = [chunk.content for chunk in chunks]

        # Create embeddings
        logger.info("Creating embeddings for chunks")
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        except Exception as e:
            logger.error("Failed to create embeddings", error=str(e))
            raise

        # Initialize or update FAISS index
        if self.index is None:
            logger.info("Creating new FAISS index", dimension=self.dimension)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        for chunk in chunks:
            metadata = DocumentMetadata(
                source=chunk.source_file,
                page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                content_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            )
            self.metadata.append(metadata)

        logger.info(
            "Documents added to vector store",
            total_vectors=self.index.ntotal if self.index else 0,
            total_metadata=len(self.metadata)
        )

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[DocumentMetadata, float]]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of (metadata, score) tuples sorted by similarity
        """
        if self.index is None or len(self.metadata) == 0:
            logger.warning("Vector store is empty, cannot perform search")
            return []

        self._load_model()

        logger.info("Searching vector store", query=query[:50], k=k, threshold=score_threshold)

        try:
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)

            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                if score < score_threshold:
                    continue

                if idx >= len(self.metadata):
                    logger.warning("Index out of range", idx=idx, metadata_len=len(self.metadata))
                    continue

                metadata = self.metadata[idx]
                metadata.similarity_score = float(score)
                results.append((metadata, float(score)))

            logger.info(
                "Search completed",
                query=query[:50],
                results_count=len(results),
                top_score=results[0][1] if results else 0.0
            )

            return results

        except Exception as e:
            logger.error("Error during vector search", error=str(e))
            raise

    def save(self, path: Path) -> None:
        """Save vector store to disk.

        Args:
            path: Directory path to save the vector store
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving vector store", path=str(path))

        try:
            # Save FAISS index
            index_path = path / "index.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            # Save configuration
            config_path = path / "config.json"
            config = {
                "model_name": self.model_name,
                "dimension": self.dimension,
                "total_vectors": self.index.ntotal,
                "metadata_count": len(self.metadata)
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("Vector store saved successfully", path=str(path))

        except Exception as e:
            logger.error("Failed to save vector store", error=str(e))
            raise

    def load(self, path: Path) -> None:
        """Load vector store from disk.

        Args:
            path: Directory path containing the vector store
        """
        if not path.exists():
            raise FileNotFoundError(f"Vector store path does not exist: {path}")

        logger.info("Loading vector store", path=str(path))

        try:
            # Load configuration
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.model_name = config.get("model_name", self.model_name)
                self.dimension = config.get("dimension")

            # Load FAISS index
            index_path = path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            else:
                raise FileNotFoundError(f"Index file not found: {index_path}")

            # Load metadata
            metadata_path = path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            logger.info(
                "Vector store loaded successfully",
                model=self.model_name,
                vectors=self.index.ntotal if self.index else 0,
                metadata_count=len(self.metadata)
            )

        except Exception as e:
            logger.error("Failed to load vector store", error=str(e))
            raise

    def exists(self, path: Path) -> bool:
        """Check if vector store exists at path.

        Args:
            path: Path to check

        Returns:
            True if vector store exists
        """
        required_files = ["index.faiss", "metadata.pkl"]
        return (
            path.exists() and
            path.is_dir() and
            all((path / file).exists() for file in required_files)
        )

    def get_stats(self) -> Dict:
        """Get vector store statistics.

        Returns:
            Dictionary with store statistics
        """
        if self.index is None:
            return {
                "total_vectors": 0,
                "metadata_count": 0,
                "model_name": self.model_name,
                "dimension": self.dimension
            }

        # Count documents and pages
        sources = set()
        pages = set()
        for metadata in self.metadata:
            sources.add(metadata.source)
            if metadata.page > 0:
                pages.add((metadata.source, metadata.page))

        return {
            "total_vectors": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "unique_sources": len(sources),
            "unique_pages": len(pages),
            "model_name": self.model_name,
            "dimension": self.dimension
        }

    def clear(self) -> None:
        """Clear the vector store."""
        logger.info("Clearing vector store")
        self.index = None
        self.metadata = []
        self.dimension = None

    def rebuild_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Rebuild vector store from document chunks.

        Args:
            chunks: List of document chunks
        """
        logger.info("Rebuilding vector store from chunks", chunk_count=len(chunks))
        self.clear()
        self.add_documents(chunks)

    def search_by_metadata(
        self,
        source: Optional[str] = None,
        page: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """Search metadata by source and/or page.

        Args:
            source: Source document name
            page: Page number

        Returns:
            List of matching metadata
        """
        results = []

        for metadata in self.metadata:
            if source is not None and metadata.source != source:
                continue
            if page is not None and metadata.page != page:
                continue
            results.append(metadata)

        return results

