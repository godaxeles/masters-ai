"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, List

from src.models.document import DocumentChunk, ProcessedDocument
from src.services.document_processor import DocumentProcessor
from src.services.vector_store import VectorStore
from src.services.chatbot import ChatBot
from src.models.chat import ChatSession


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    content = """This is a sample document for testing purposes.
It contains multiple sentences and paragraphs.

This is the second paragraph with more content.
We want to test document processing capabilities.
The content should be split into appropriate chunks.

This is the third paragraph for testing overlap functionality.
Each chunk should maintain context while staying within size limits.
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_file(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        file_path = temp_dir / "sample.pdf"

        c = canvas.Canvas(str(file_path), pagesize=letter)
        c.drawString(100, 750, "Sample PDF Document")
        c.drawString(100, 730, "This is page 1 content for testing.")
        c.drawString(100, 710, "It contains sample text to test PDF processing.")
        c.showPage()

        c.drawString(100, 750, "Page 2 Content")
        c.drawString(100, 730, "This is the second page of the PDF document.")
        c.drawString(100, 710, "Used for testing multi-page PDF processing.")
        c.save()

        return file_path
    except ImportError:
        # Skip PDF tests if reportlab not available
        pytest.skip("reportlab not available for PDF testing")


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            source_file="test_doc.txt",
            page_number=1,
            content="This is the first chunk of test content. It contains information about testing.",
            metadata={"source": "test_doc.txt", "page": 1, "chunk_id": "chunk_1"}
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            source_file="test_doc.txt",
            page_number=1,
            content="This is the second chunk with different content. It discusses document processing.",
            metadata={"source": "test_doc.txt", "page": 1, "chunk_id": "chunk_2"}
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            source_file="another_doc.txt",
            page_number=1,
            content="This chunk is from a different document. It covers vector store functionality.",
            metadata={"source": "another_doc.txt", "page": 1, "chunk_id": "chunk_3"}
        )
    ]


@pytest.fixture
def sample_processed_document(sample_chunks: List[DocumentChunk]) -> ProcessedDocument:
    """Create a sample processed document for testing."""
    return ProcessedDocument(
        file_path=Path("test_doc.txt"),
        file_name="test_doc.txt",
        file_size=1024,
        file_type="txt",
        total_pages=1,
        total_chunks=len(sample_chunks),
        chunks=sample_chunks[:2]  # First two chunks belong to this document
    )


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """Create a document processor instance."""
    return DocumentProcessor()


@pytest.fixture
def vector_store(temp_dir: Path) -> VectorStore:
    """Create a vector store instance."""
    return VectorStore()


@pytest.fixture
def populated_vector_store(vector_store: VectorStore, sample_chunks: List[DocumentChunk]) -> VectorStore:
    """Create a vector store populated with sample data."""
    vector_store.add_documents(sample_chunks)
    return vector_store


@pytest.fixture
def chat_session() -> ChatSession:
    """Create a sample chat session."""
    return ChatSession(
        session_id="test_session_123",
        user_id="test_user"
    )


@pytest.fixture
def chatbot(populated_vector_store: VectorStore) -> ChatBot:
    """Create a chatbot instance with populated vector store."""
    chatbot = ChatBot()
    chatbot.vector_store = populated_vector_store
    return chatbot


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "rag: RAG-specific tests")


# Test collection filters
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark RAG-related tests
        if any(keyword in str(item.fspath) for keyword in ["vector_store", "chatbot", "rag"]):
            item.add_marker(pytest.mark.rag)

