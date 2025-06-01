"""Unit tests for vector store."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.services.vector_store import VectorStore
from src.models.document import DocumentChunk, DocumentMetadata


@pytest.mark.unit
@pytest.mark.rag
class TestVectorStore:
    """Test vector store functionality."""

    def test_init(self):
        """Test vector store initialization."""
        vs = VectorStore()
        assert vs.model is None
        assert vs.index is None
        assert vs.metadata == []
        assert vs.dimension is None

    @patch('src.services.vector_store.SentenceTransformer')
    def test_load_model(self, mock_transformer):
        """Test model loading."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model

        vs = VectorStore()
        vs._load_model()

        assert vs.model is not None
        assert vs.dimension == 3
        mock_transformer.assert_called_once()

    @patch('src.services.vector_store.SentenceTransformer')
    def test_add_documents(self, mock_transformer, sample_chunks):
        """Test adding documents to vector store."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_transformer.return_value = mock_model

        vs = VectorStore()
        vs.add_documents(sample_chunks)

        assert vs.index is not None
        assert vs.index.ntotal == len(sample_chunks)
        assert len(vs.metadata) == len(sample_chunks)

        # Check metadata
        for i, chunk in enumerate(sample_chunks):
            metadata = vs.metadata[i]
            assert metadata.source == chunk.source_file
            assert metadata.page == chunk.page_number
            assert metadata.chunk_id == chunk.chunk_id

    def test_add_empty_documents(self, vector_store):
        """Test adding empty document list."""
        vector_store.add_documents([])
        assert vector_store.index is None
        assert len(vector_store.metadata) == 0

    @patch('src.services.vector_store.SentenceTransformer')
    def test_search(self, mock_transformer, sample_chunks):
        """Test searching vector store."""
        # Mock the model
        mock_model = MagicMock()
        # Mock training embeddings
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),  # Training
            np.array([[0.15, 0.25, 0.35]])  # Query
        ]
        mock_transformer.return_value = mock_model

        vs = VectorStore()
        vs.add_documents(sample_chunks)

        results = vs.search("test query", k=2)

        assert len(results) <= 2
        assert all(isinstance(metadata, DocumentMetadata) for metadata, score in results)
        assert all(isinstance(score, float) for metadata, score in results)
        assert all(0 <= score <= 1 for metadata, score in results)

    def test_search_empty_store(self, vector_store):
        """Test searching empty vector store."""
        results = vector_store.search("test query")
        assert len(results) == 0

    @patch('src.services.vector_store.SentenceTransformer')
    def test_search_with_threshold(self, mock_transformer, sample_chunks):
        """Test searching with similarity threshold."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),  # Training
            np.array([[0.0, 0.0, 0.0]])  # Query that should have low similarity
        ]
        mock_transformer.return_value = mock_model

        vs = VectorStore()
        vs.add_documents(sample_chunks)

        # High threshold should return fewer results
        results_high = vs.search("test query", score_threshold=0.9)
        results_low = vs.search("test query", score_threshold=0.1)

        assert len(results_high) <= len(results_low)

    def test_save_load(self, vector_store, sample_chunks, temp_dir):
        """Test saving and loading vector store."""
        # Skip if we can't load the model
        pytest.skip("Requires model loading for full integration test")

    def test_exists(self, temp_dir):
        """Test checking if vector store exists."""
        vs = VectorStore()

        # Should not exist in empty directory
        assert not vs.exists(temp_dir)

        # Create required files
        (temp_dir / "index.faiss").touch()
        (temp_dir / "metadata.pkl").touch()

        # Should exist now
        assert vs.exists(temp_dir)

    def test_get_stats(self, vector_store):
        """Test getting vector store statistics."""
        stats = vector_store.get_stats()

        assert isinstance(stats, dict)
        assert "total_vectors" in stats
        assert "metadata_count" in stats
        assert "model_name" in stats
        assert "dimension" in stats

        # Empty store stats
        assert stats["total_vectors"] == 0
        assert stats["metadata_count"] == 0

    def test_clear(self, vector_store):
        """Test clearing vector store."""
        vector_store.clear()

        assert vector_store.index is None
        assert vector_store.metadata == []
        assert vector_store.dimension is None

    def test_search_by_metadata(self, populated_vector_store):
        """Test searching by metadata."""
        # Search by source
        results = populated_vector_store.search_by_metadata(source="test_doc.txt")
        assert len(results) == 2  # Two chunks from test_doc.txt

        # Search by source and page
        results = populated_vector_store.search_by_metadata(source="test_doc.txt", page=1)
        assert len(results) == 2

        # Search for non-existent source
        results = populated_vector_store.search_by_metadata(source="nonexistent.txt")
        assert len(results) == 0

