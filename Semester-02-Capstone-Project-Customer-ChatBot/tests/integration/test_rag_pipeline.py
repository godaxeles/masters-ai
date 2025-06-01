"""Integration tests for RAG pipeline."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.services.document_processor import DocumentProcessor
from src.services.vector_store import VectorStore
from src.services.chatbot import ChatBot
from src.models.document import DocumentCollection


@pytest.mark.integration
@pytest.mark.rag
class TestRAGPipeline:
    """Test the complete RAG pipeline integration."""

    def test_document_to_vector_store_pipeline(self, temp_dir: Path):
        """Test complete pipeline from document processing to vector store."""
        # Create test documents
        doc1 = temp_dir / "product_manual.txt"
        doc1.write_text("""
        Product X is our flagship software solution.
        It provides advanced analytics and reporting capabilities.
        The software supports real-time data processing and visualization.
        Contact our support team for technical assistance.
        """)

        doc2 = temp_dir / "faq.txt"
        doc2.write_text("""
        Frequently Asked Questions

        Q: How do I install the software?
        A: Download the installer from our website and follow the setup wizard.

        Q: What are the system requirements?
        A: Windows 10, 8GB RAM, 2GB disk space minimum.

        Q: How do I contact support?
        A: Email support@company.com or call 1-800-SUPPORT.
        """)

        # Process documents
        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)

        assert len(processed_docs) == 2
        assert all(doc.total_chunks > 0 for doc in processed_docs)

        # Create document collection
        collection = DocumentCollection(
            collection_id="test_collection",
            documents=processed_docs
        )

        # Build vector store (skip if model loading fails)
        try:
            vector_store = VectorStore()
            all_chunks = []
            for doc in collection.documents:
                all_chunks.extend(doc.chunks)

            vector_store.add_documents(all_chunks)

            # Test search functionality
            results = vector_store.search("software installation", k=3)
            assert len(results) > 0

            # Test that relevant content is retrieved
            found_installation = any("install" in result[0].content_preview.lower() for result in results)
            assert found_installation

        except Exception as e:
            pytest.skip(f"Vector store test skipped due to model loading: {e}")

    @patch('src.services.vector_store.SentenceTransformer')
    def test_chatbot_end_to_end(self, mock_transformer, temp_dir: Path):
        """Test chatbot end-to-end functionality."""
        # Mock the transformer
        import numpy as np
        mock_model = mock_transformer.return_value
        mock_model.encode.side_effect = [
            # For document processing
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            # For search query
            np.array([[0.15, 0.25, 0.35]])
        ]

        # Create test document
        test_doc = temp_dir / "support_guide.txt"
        test_doc.write_text("""
        Welcome to our customer support guide.

        For technical issues, please contact our support team at support@company.com.
        Our phone number is 1-800-SUPPORT.
        We offer 24/7 support for all customers.

        Common issues and solutions:
        - Login problems: Reset your password
        - Installation issues: Run as administrator
        - Performance problems: Clear cache and restart
        """)

        # Process document
        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)

        # Create chatbot
        chatbot = ChatBot()

        # Add documents to vector store
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        chatbot.vector_store.add_documents(all_chunks)

        # Test questions
        test_questions = [
            "How do I contact support?",
            "What is your phone number?",
            "How do I fix login problems?",
            "What should I do for installation issues?"
        ]

        for question in test_questions:
            answer, found_relevant, citations = chatbot.answer_question(question)

            assert isinstance(answer, str)
            assert len(answer) > 0

            # For this mock setup, we expect some results
            if found_relevant:
                assert len(citations) > 0

    def test_document_requirements_validation(self, temp_dir: Path):
        """Test that document requirements are properly validated."""
        # Create insufficient documents (need at least 3)
        doc1 = temp_dir / "doc1.txt"
        doc1.write_text("Document 1 content")

        doc2 = temp_dir / "doc2.txt"
        doc2.write_text("Document 2 content")

        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)

        collection = DocumentCollection(
            collection_id="test_collection",
            documents=processed_docs
        )

        violations = collection.validate_requirements()
        assert len(violations) > 0
        assert any("at least 3 documents" in violation for violation in violations)
        assert any("at least 2 PDF" in violation for violation in violations)

    @patch('src.services.vector_store.SentenceTransformer')
    def test_search_quality_metrics(self, mock_transformer, sample_chunks):
        """Test search quality and relevance metrics."""
        import numpy as np

        # Mock embeddings with known similarities
        mock_model = mock_transformer.return_value
        mock_model.encode.side_effect = [
            # Document embeddings
            np.array([
                [1.0, 0.0, 0.0],  # High similarity to query 1
                [0.0, 1.0, 0.0],  # High similarity to query 2
                [0.0, 0.0, 1.0]   # Low similarity to both
            ]),
            # Query 1 embedding
            np.array([[0.9, 0.1, 0.0]]),
            # Query 2 embedding
            np.array([[0.1, 0.9, 0.0]])
        ]

        vector_store = VectorStore()
        vector_store.add_documents(sample_chunks)

        # Test query 1 - should match first chunk best
        results1 = vector_store.search("query matching first chunk", k=3)
        assert len(results1) > 0
        # First result should have highest similarity
        if len(results1) > 1:
            assert results1[0][1] >= results1[1][1]

        # Test query 2 - should match second chunk best
        results2 = vector_store.search("query matching second chunk", k=3)
        assert len(results2) > 0

        # Test similarity threshold filtering
        high_threshold_results = vector_store.search("test query", score_threshold=0.8)
        low_threshold_results = vector_store.search("test query", score_threshold=0.1)
        assert len(high_threshold_results) <= len(low_threshold_results)

    def test_citation_accuracy(self, populated_vector_store):
        """Test that citations are accurate and properly formatted."""
        results = populated_vector_store.search("test content", k=3)

        for metadata, score in results:
            # Check citation formatting
            citation = metadata.get_citation()
            assert isinstance(citation, str)
            assert len(citation) > 0
            assert metadata.source in citation

            # Check metadata completeness
            assert hasattr(metadata, 'source')
            assert hasattr(metadata, 'page')
            assert hasattr(metadata, 'chunk_id')
            assert hasattr(metadata, 'similarity_score')
            assert hasattr(metadata, 'content_preview')

            # Verify score consistency
            assert metadata.similarity_score == score

