"""Document data models."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of processed document content."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    source_file: str = Field(..., description="Original source file name")
    page_number: int = Field(default=0, description="Page number (0 for non-paginated docs)")
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    word_count: int = Field(default=0, description="Number of words in chunk")
    char_count: int = Field(default=0, description="Number of characters in chunk")

    def __post_init__(self):
        """Calculate word and character counts."""
        if self.content:
            self.word_count = len(self.content.split())
            self.char_count = len(self.content)


class ProcessedDocument(BaseModel):
    """A processed document with its chunks."""

    file_path: Path = Field(..., description="Path to the original file")
    file_name: str = Field(..., description="Name of the file")
    file_size: int = Field(..., description="Size of file in bytes")
    file_type: str = Field(..., description="Type of file (pdf, txt, etc.)")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    total_pages: int = Field(default=0, description="Total pages in document")
    total_chunks: int = Field(default=0, description="Total chunks created")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    processing_errors: List[str] = Field(default_factory=list, description="Any processing errors")

    @property
    def total_words(self) -> int:
        """Get total word count across all chunks."""
        return sum(chunk.word_count for chunk in self.chunks)

    @property
    def total_characters(self) -> int:
        """Get total character count across all chunks."""
        return sum(chunk.char_count for chunk in self.chunks)

    @property
    def is_large_document(self) -> bool:
        """Check if this is a large document (400+ pages or 10MB+)."""
        return self.total_pages >= 400 or self.file_size >= 10 * 1024 * 1024


class DocumentMetadata(BaseModel):
    """Metadata for document retrieval and citation."""

    source: str = Field(..., description="Source document name")
    page: int = Field(default=0, description="Page number")
    chunk_id: str = Field(..., description="Chunk identifier")
    similarity_score: float = Field(default=0.0, description="Similarity score from retrieval")
    content_preview: str = Field(default="", description="Preview of content")

    def get_citation(self) -> str:
        """Generate a citation string for this document.

        Returns:
            Formatted citation string
        """
        if self.page > 0:
            return f"{self.source} (page {self.page})"
        else:
            return f"{self.source}"


class DocumentCollection(BaseModel):
    """Collection of processed documents."""

    collection_id: str = Field(..., description="Unique collection identifier")
    name: str = Field(default="Document Collection", description="Collection name")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    documents: List[ProcessedDocument] = Field(default_factory=list, description="Documents in collection")

    @property
    def total_documents(self) -> int:
        """Get total number of documents."""
        return len(self.documents)

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks across all documents."""
        return sum(doc.total_chunks for doc in self.documents)

    @property
    def total_pages(self) -> int:
        """Get total pages across all documents."""
        return sum(doc.total_pages for doc in self.documents)

    @property
    def pdf_documents(self) -> List[ProcessedDocument]:
        """Get only PDF documents."""
        return [doc for doc in self.documents if doc.file_type.lower() == 'pdf']

    @property
    def large_documents(self) -> List[ProcessedDocument]:
        """Get documents with 400+ pages."""
        return [doc for doc in self.documents if doc.is_large_document]

    def get_document_by_name(self, name: str) -> Optional[ProcessedDocument]:
        """Get document by file name.

        Args:
            name: File name to search for

        Returns:
            Document if found, None otherwise
        """
        for doc in self.documents:
            if doc.file_name == name:
                return doc
        return None

    def add_document(self, document: ProcessedDocument) -> None:
        """Add a document to the collection.

        Args:
            document: Document to add
        """
        self.documents.append(document)
        self.updated_at = datetime.now()

    def remove_document(self, file_name: str) -> bool:
        """Remove a document from the collection.

        Args:
            file_name: Name of file to remove

        Returns:
            True if document was removed, False if not found
        """
        for i, doc in enumerate(self.documents):
            if doc.file_name == file_name:
                del self.documents[i]
                self.updated_at = datetime.now()
                return True
        return False

    def validate_requirements(self) -> List[str]:
        """Validate that collection meets project requirements.

        Returns:
            List of requirement violations (empty if all requirements met)
        """
        violations = []

        # At least 3 documents
        if self.total_documents < 3:
            violations.append(f"Need at least 3 documents, have {self.total_documents}")

        # At least 2 PDFs
        pdf_count = len(self.pdf_documents)
        if pdf_count < 2:
            violations.append(f"Need at least 2 PDF documents, have {pdf_count}")

        # At least 1 large document
        large_count = len(self.large_documents)
        if large_count < 1:
            violations.append(f"Need at least 1 document with 400+ pages, have {large_count}")

        return violations