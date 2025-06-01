"""Test data fixtures and utilities."""

from typing import List, Dict, Any
from datetime import datetime, timedelta

from src.models.document import DocumentChunk, ProcessedDocument
from src.models.chat import ChatMessage, ChatSession, MessageRole, Citation
from src.models.ticket import SupportTicket, TicketRequest, TicketPriority, TicketCategory


def create_sample_chat_history() -> List[ChatMessage]:
    """Create sample chat history for testing."""
    return [
        ChatMessage(
            message_id="msg_001",
            role=MessageRole.USER,
            content="Hello, I need help with your product",
            timestamp=datetime.now() - timedelta(minutes=10)
        ),
        ChatMessage(
            message_id="msg_002",
            role=MessageRole.ASSISTANT,
            content="Hello! I'd be happy to help you with our product. What specific question do you have?",
            timestamp=datetime.now() - timedelta(minutes=9),
            processing_time=1.2
        ),
        ChatMessage(
            message_id="msg_003",
            role=MessageRole.USER,
            content="How do I reset my password?",
            timestamp=datetime.now() - timedelta(minutes=8)
        ),
        ChatMessage(
            message_id="msg_004",
            role=MessageRole.ASSISTANT,
            content="To reset your password, please follow these steps:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Enter your email address\n4. Check your email for reset instructions",
            timestamp=datetime.now() - timedelta(minutes=7),
            processing_time=0.8,
            citations=[
                Citation(
                    source="user_guide.pdf",
                    page=12,
                    chunk_id="chunk_password_reset",
                    similarity_score=0.92,
                    excerpt="Password reset instructions can be found..."
                )
            ]
        )
    ]


def create_sample_support_tickets() -> List[SupportTicket]:
    """Create sample support tickets for testing."""
    return [
        SupportTicket(
            ticket_id="TICKET_001",
            title="Cannot login to account",
            description="I'm unable to login to my account. I've tried resetting my password but still can't get in.",
            user_name="John Smith",
            user_email="john.smith@email.com",
            priority=TicketPriority.HIGH,
            category=TicketCategory.ACCOUNT_ISSUE,
            created_at=datetime.now() - timedelta(hours=2),
            external_id="123",
            external_url="https://github.com/company/support/issues/123"
        ),
        SupportTicket(
            ticket_id="TICKET_002",
            title="Feature request: Dark mode",
            description="It would be great if the application supported a dark mode theme for better usability in low-light conditions.",
            user_name="Jane Doe",
            user_email="jane.doe@email.com",
            priority=TicketPriority.LOW,
            category=TicketCategory.FEATURE_REQUEST,
            created_at=datetime.now() - timedelta(days=1),
            external_id="124",
            external_url="https://github.com/company/support/issues/124"
        ),
        SupportTicket(
            ticket_id="TICKET_003",
            title="Application crashes on startup",
            description="The application crashes immediately when I try to start it. This started happening after the latest update.",
            user_name="Bob Wilson",
            user_email="bob.wilson@email.com",
            priority=TicketPriority.URGENT,
            category=TicketCategory.BUG_REPORT,
            created_at=datetime.now() - timedelta(minutes=30),
            external_id="125",
            external_url="https://github.com/company/support/issues/125"
        )
    ]


def create_sample_documents() -> List[ProcessedDocument]:
    """Create sample processed documents for testing."""
    # User Guide Document
    user_guide_chunks = [
        DocumentChunk(
            chunk_id="ug_chunk_001",
            source_file="user_guide.pdf",
            page_number=1,
            content="Welcome to our application! This user guide will help you get started with all the features and functionality.",
            metadata={"source": "user_guide.pdf", "page": 1, "chunk_id": "ug_chunk_001"}
        ),
        DocumentChunk(
            chunk_id="ug_chunk_002",
            source_file="user_guide.pdf",
            page_number=5,
            content="To create a new account, click the 'Sign Up' button on the homepage. Enter your email address, choose a strong password, and verify your email.",
            metadata={"source": "user_guide.pdf", "page": 5, "chunk_id": "ug_chunk_002"}
        ),
        DocumentChunk(
            chunk_id="ug_chunk_003",
            source_file="user_guide.pdf",
            page_number=12,
            content="Password reset instructions: If you forget your password, use the 'Forgot Password' link on the login page. Enter your email address and follow the instructions sent to your email.",
            metadata={"source": "user_guide.pdf", "page": 12, "chunk_id": "ug_chunk_003"}
        )
    ]

    # FAQ Document
    faq_chunks = [
        DocumentChunk(
            chunk_id="faq_chunk_001",
            source_file="faq.txt",
            page_number=0,
            content="Q: How do I change my subscription plan? A: You can change your subscription plan by going to Account Settings > Billing > Change Plan.",
            metadata={"source": "faq.txt", "page": 0, "chunk_id": "faq_chunk_001"}
        ),
        DocumentChunk(
            chunk_id="faq_chunk_002",
            source_file="faq.txt",
            page_number=0,
            content="Q: Is there a mobile app available? A: Yes, our mobile app is available for both iOS and Android devices. Download it from the App Store or Google Play.",
            metadata={"source": "faq.txt", "page": 0, "chunk_id": "faq_chunk_002"}
        )
    ]

    # API Documentation
    api_chunks = [
        DocumentChunk(
            chunk_id="api_chunk_001",
            source_file="api_documentation.pdf",
            page_number=3,
            content="Authentication: All API requests must include an API key in the Authorization header. Format: Authorization: Bearer YOUR_API_KEY",
            metadata={"source": "api_documentation.pdf", "page": 3, "chunk_id": "api_chunk_001"}
        ),
        DocumentChunk(
            chunk_id="api_chunk_002",
            source_file="api_documentation.pdf",
            page_number=15,
            content="Rate limiting: API requests are limited to 1000 requests per hour per API key. Exceeding this limit will result in HTTP 429 responses.",
            metadata={"source": "api_documentation.pdf", "page": 15, "chunk_id": "api_chunk_002"}
        )
    ]

    return [
        ProcessedDocument(
            file_path="data/raw/user_guide.pdf",
            file_name="user_guide.pdf",
            file_size=2048000,  # 2MB
            file_type="pdf",
            total_pages=50,
            total_chunks=len(user_guide_chunks),
            chunks=user_guide_chunks
        ),
        ProcessedDocument(
            file_path="data/raw/faq.txt",
            file_name="faq.txt",
            file_size=15000,  # 15KB
            file_type="txt",
            total_pages=1,
            total_chunks=len(faq_chunks),
            chunks=faq_chunks
        ),
        ProcessedDocument(
            file_path="data/raw/api_documentation.pdf",
            file_name="api_documentation.pdf",
            file_size=5120000,  # 5MB
            file_type="pdf",
            total_pages=100,
            total_chunks=len(api_chunks),
            chunks=api_chunks
        )
    ]


def create_performance_test_data() -> Dict[str, Any]:
    """Create data for performance testing."""
    return {
        "large_document_chunks": [
            DocumentChunk(
                chunk_id=f"perf_chunk_{i:04d}",
                source_file="large_manual.pdf",
                page_number=(i // 10) + 1,
                content=f"This is performance test content chunk {i}. " * 50,  # ~300 words
                metadata={"source": "large_manual.pdf", "page": (i // 10) + 1, "chunk_id": f"perf_chunk_{i:04d}"}
            )
            for i in range(1000)  # 1000 chunks
        ],
        "test_queries": [
            "How do I configure the system?",
            "What are the installation requirements?",
            "How to troubleshoot connection issues?",
            "What is the API rate limit?",
            "How to reset user passwords?",
            "What payment methods are supported?",
            "How to export data?",
            "What are the security features?",
            "How to set up integrations?",
            "What is the refund policy?"
        ] * 10,  # 100 queries total
        "expected_response_time": 2.0,  # seconds
        "expected_accuracy_threshold": 0.7
    }


def create_edge_case_test_data() -> Dict[str, Any]:
    """Create edge case test data."""
    return {
        "empty_queries": ["", "   ", "\n\t", "?", "!"],
        "very_long_query": "What is " + "very " * 200 + "long query?",
        "special_characters": "How do I use @#$%^&*(){}[]|\\:;\"'<>?/~`",
        "multilingual": "¿Cómo puedo hacer esto? Comment puis-je faire cela? 这怎么做？",
        "code_injection": "<script>alert('test')</script>",
        "sql_injection": "'; DROP TABLE users; --",
        "very_short_chunks": [
            DocumentChunk(
                chunk_id="short_001",
                source_file="short.txt",
                page_number=1,
                content="Hi.",
                metadata={"source": "short.txt", "page": 1, "chunk_id": "short_001"}
            )
        ],
        "empty_chunks": [
            DocumentChunk(
                chunk_id="empty_001",
                source_file="empty.txt",
                page_number=1,
                content="",
                metadata={"source": "empty.txt", "page": 1, "chunk_id": "empty_001"}
            )
        ]
    }

