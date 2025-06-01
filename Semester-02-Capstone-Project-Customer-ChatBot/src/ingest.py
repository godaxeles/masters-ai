"""Document ingestion script for processing and indexing documents."""

import argparse
import sys
from pathlib import Path
from typing import List

import structlog

from config import settings
from models.document import DocumentCollection, ProcessedDocument
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from utils.logging_config import setup_logging, get_logger
from utils.validators import validate_documents_directory, validate_configuration

# Setup logging
setup_logging(log_file=settings.logs_path / "ingest.log")
logger = get_logger(__name__)


def process_documents(data_path: Path, force_reprocess: bool = False) -> DocumentCollection:
    """Process all documents in the data directory.

    Args:
        data_path: Path to documents directory
        force_reprocess: Whether to force reprocessing even if vector store exists

    Returns:
        DocumentCollection with processed documents
    """
    logger.info("Starting document processing", data_path=str(data_path))

    # Validate documents directory
    is_valid, issues = validate_documents_directory(data_path)
    if not is_valid:
        logger.error("Document directory validation failed", issues=issues)
        for issue in issues:
            print(f"❌ {issue}")
        sys.exit(1)

    # Initialize processor
    processor = DocumentProcessor()

    try:
        # Process all documents
        processed_docs = processor.process_directory(data_path)

        if not processed_docs:
            logger.error("No documents were processed successfully")
            sys.exit(1)

        # Create document collection
        collection = DocumentCollection(
            collection_id="main_collection",
            name="Customer Support Documents",
            documents=processed_docs
        )

        # Validate requirements
        violations = collection.validate_requirements()
        if violations:
            logger.error("Document collection requirements not met", violations=violations)
            for violation in violations:
                print(f"❌ {violation}")
            sys.exit(1)

        logger.info(
            "Document processing completed",
            total_documents=collection.total_documents,
            total_chunks=collection.total_chunks,
            total_pages=collection.total_pages,
            pdf_documents=len(collection.pdf_documents),
            large_documents=len(collection.large_documents)
        )

        # Print summary
        print("\n📄 Document Processing Summary:")
        print(f"   Total Documents: {collection.total_documents}")
        print(f"   Total Chunks: {collection.total_chunks}")
        print(f"   Total Pages: {collection.total_pages}")
        print(f"   PDF Documents: {len(collection.pdf_documents)}")
        print(f"   Large Documents (400+ pages): {len(collection.large_documents)}")

        return collection

    except Exception as e:
        logger.error("Document processing failed", error=str(e))
        print(f"❌ Document processing failed: {e}")
        sys.exit(1)


def create_vector_store(collection: DocumentCollection, store_path: Path, force_rebuild: bool = False) -> None:
    """Create and save vector store from document collection.

    Args:
        collection: Processed document collection
        store_path: Path to save vector store
        force_rebuild: Whether to force rebuilding existing vector store
    """
    logger.info("Creating vector store", store_path=str(store_path))

    # Check if vector store already exists
    vector_store = VectorStore()
    if vector_store.exists(store_path) and not force_rebuild:
        logger.info("Vector store already exists, skipping creation")
        print("✅ Vector store already exists (use --force-rebuild to recreate)")
        return

    try:
        # Collect all chunks from all documents
        all_chunks = []
        for doc in collection.documents:
            all_chunks.extend(doc.chunks)

        if not all_chunks:
            logger.error("No chunks found in document collection")
            print("❌ No document chunks found to index")
            sys.exit(1)

        # Create embeddings and build vector store
        print(f"🔄 Creating embeddings for {len(all_chunks)} chunks...")
        vector_store.add_documents(all_chunks)

        # Save vector store
        print(f"💾 Saving vector store to {store_path}...")
        vector_store.save(store_path)

        # Get and print statistics
        stats = vector_store.get_stats()
        logger.info("Vector store created successfully", **stats)

        print("\n🗄️  Vector Store Created:")
        print(f"   Total Vectors: {stats['total_vectors']}")
        print(f"   Unique Sources: {stats['unique_sources']}")
        print(f"   Unique Pages: {stats['unique_pages']}")
        print(f"   Model: {stats['model_name']}")
        print(f"   Dimension: {stats['dimension']}")

    except Exception as e:
        logger.error("Vector store creation failed", error=str(e))
        print(f"❌ Vector store creation failed: {e}")
        sys.exit(1)


def main():
    """Main ingestion script."""
    parser = argparse.ArgumentParser(description="Process documents and create vector store")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_raw_path,
        help="Path to documents directory"
    )
    parser.add_argument(
        "--vector-store-path",
        type=Path,
        default=settings.vector_store_path,
        help="Path to save vector store"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of documents"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding of vector store"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and documents"
    )

    args = parser.parse_args()

    print("🚀 Customer Support RAG Chatbot - Document Ingestion")
    print("=" * 60)

    # Validate configuration
    print("🔧 Validating configuration...")
    is_valid, issues = validate_configuration()
    if not is_valid:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        sys.exit(1)
    print("✅ Configuration valid")

    # Validate documents
    print(f"📁 Validating documents in {args.data_path}...")
    is_valid, issues = validate_documents_directory(args.data_path)
    if not is_valid:
        print("❌ Document validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        sys.exit(1)
    print("✅ Documents valid")

    if args.validate_only:
        print("\n✅ Validation completed successfully!")
        return

    # Process documents
    collection = process_documents(args.data_path, args.force_reprocess)

    # Create vector store
    create_vector_store(collection, args.vector_store_path, args.force_rebuild)

    print("\n🎉 Document ingestion completed successfully!")
    print(f"📚 Processed {collection.total_documents} documents with {collection.total_chunks} chunks")
    print(f"🗄️  Vector store saved to {args.vector_store_path}")
    print("\n🚀 You can now run the application with: streamlit run src/app.py")


if __name__ == "__main__":
    main()

