"""End-to-end tests for complete user workflows."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from src.services.document_processor import DocumentProcessor
from src.services.vector_store import VectorStore
from src.services.chatbot import ChatBot
from src.services.ticket_service import TicketService
from src.models.chat import ChatSession, MessageRole
from src.models.ticket import TicketRequest


@pytest.mark.e2e
class TestUserFlows:
    """Test complete user workflows end-to-end."""

    @patch('src.services.vector_store.SentenceTransformer')
    @patch('requests.post')
    def test_complete_support_workflow(self, mock_post, mock_transformer, temp_dir: Path):
        """Test complete user support workflow from question to ticket."""
        # Setup mocks
        mock_model = mock_transformer.return_value
        mock_model.encode.side_effect = [
            # Document processing
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            # Search queries (2 searches)
            np.array([[0.0, 0.0, 0.1]]),  # Low similarity - no relevant answer
            np.array([[0.0, 0.0, 0.1]])   # Second search also low similarity
        ]

        # Mock GitHub API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'number': 789,
            'html_url': 'https://github.com/test/repo/issues/789'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Step 1: Create knowledge base
        support_doc = temp_dir / "support_guide.txt"
        support_doc.write_text("""
        Customer Support Guide

        For billing questions, contact billing@company.com
        For technical support, call 1-800-TECH-HELP
        Business hours: Monday-Friday 9AM-5PM EST

        Common Issues:
        - Password reset: Use the forgot password link
        - Account locked: Contact support with your email
        - Payment issues: Check your payment method
        """)

        faq_doc = temp_dir / "faq.txt"
        faq_doc.write_text("""
        Frequently Asked Questions

        Q: How do I change my subscription?
        A: Log into your account and go to billing settings.

        Q: Can I get a refund?
        A: Refunds are available within 30 days of purchase.

        Q: Is there a mobile app?
        A: Yes, download from App Store or Google Play.
        """)

        # Step 2: Process documents
        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)
        assert len(processed_docs) == 2

        # Step 3: Build vector store
        vector_store = VectorStore()
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        vector_store.add_documents(all_chunks)

        # Step 4: Initialize chatbot
        chatbot = ChatBot()
        chatbot.vector_store = vector_store

        # Step 5: User session starts
        session = ChatSession(session_id="test_session", user_id="test_user")

        # Step 6: User asks questions
        questions_and_expectations = [
            ("How do I reset my password?", True),  # Should find answer
            ("What are your business hours?", True),  # Should find answer
            ("How do I integrate with Salesforce?", False),  # Should not find answer
        ]

        for question, expect_answer in questions_and_expectations:
            # User asks question
            user_message = chatbot.create_chat_message(
                content=question,
                role=MessageRole.USER
            )
            session.add_message(user_message)

            # Chatbot responds
            answer, found_relevant, citations = chatbot.answer_question(
                question, session=session
            )

            assistant_message = chatbot.create_chat_message(
                content=answer,
                role=MessageRole.ASSISTANT,
                citations=citations
            )
            session.add_message(assistant_message)

            # Verify response
            assert isinstance(answer, str)
            assert len(answer) > 0

        # Step 7: User needs to create ticket for unanswered question
        ticket_service = TicketService()

        # User creates support ticket
        ticket_request = TicketRequest(
            title="Need help with Salesforce integration",
            description="I need assistance setting up integration between your platform and Salesforce CRM. The documentation doesn't cover this specific use case.",
            user_name="John Doe",
            user_email="john.doe@company.com",
            original_query="How do I integrate with Salesforce?",
            chat_context=session.get_conversation_context()
        )

        ticket = ticket_service.create_ticket(ticket_request)

        # Verify ticket creation
        assert ticket.external_id == "789"
        assert ticket.external_url == 'https://github.com/test/repo/issues/789'
        assert ticket.original_query == "How do I integrate with Salesforce?"
        assert len(ticket.chat_context) > 0

        # Update session
        session.tickets_created += 1

        # Step 8: Verify complete workflow metrics
        assert session.message_count >= 6  # At least 3 questions + 3 responses
        assert session.tickets_created == 1
        assert session.duration_minutes >= 0

    @patch('src.services.vector_store.SentenceTransformer')
    def test_successful_support_resolution(self, mock_transformer, temp_dir: Path):
        """Test workflow where user gets satisfactory answer without ticket."""
        # Setup mock with high similarity for relevant answer
        mock_model = mock_transformer.return_value
        mock_model.encode.side_effect = [
            # Document processing
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            # Search query with high similarity
            np.array([[0.15, 0.25, 0.35]])
        ]

        # Create relevant documentation
        help_doc = temp_dir / "user_guide.txt"
        help_doc.write_text("""
        User Guide

        Account Management:
        To change your password, follow these steps:
        1. Log into your account
        2. Click on Profile Settings
        3. Select Change Password
        4. Enter your current password
        5. Enter your new password twice
        6. Click Save Changes

        Your password must be at least 8 characters long and contain:
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character
        """)

        # Process and setup
        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)

        vector_store = VectorStore()
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        vector_store.add_documents(all_chunks)

        chatbot = ChatBot()
        chatbot.vector_store = vector_store

        # User asks question
        answer, found_relevant, citations = chatbot.answer_question(
            "How do I change my password?"
        )

        # Verify successful resolution
        assert found_relevant is True
        assert len(citations) > 0
        assert "password" in answer.lower()
        assert "log into" in answer.lower() or "login" in answer.lower()

        # Verify citation quality
        for citation in citations:
            assert citation.source == "user_guide.txt"
            assert citation.similarity_score > 0
            assert len(citation.excerpt) > 0

    def test_company_information_queries(self, temp_dir: Path):
        """Test queries about company information."""
        chatbot = ChatBot()

        company_questions = [
            "What is your contact information?",
            "How can I reach customer support?",
            "What is your phone number?",
            "What is your email address?",
            "Where is your company located?"
        ]

        for question in company_questions:
            answer, found_relevant, citations = chatbot.answer_question(question)

            # Should provide company information
            assert isinstance(answer, str)
            assert len(answer) > 0

            # Should contain company details
            company_info = chatbot.company_info.get_company_info()
            # At least some company info should be in the response
            assert any(
                info_part.lower() in answer.lower()
                for info_part in ["email", "phone", "address", "website"]
            )

    @patch('src.services.vector_store.SentenceTransformer')
    def test_conversation_context_handling(self, mock_transformer, temp_dir: Path):
        """Test that conversation context is properly maintained."""
        # Setup mock
        mock_model = mock_transformer.return_value
        mock_model.encode.side_effect = [
            # Document processing
            np.array([[0.1, 0.2, 0.3]]),
            # Multiple search queries
            np.array([[0.2, 0.3, 0.4]]),
            np.array([[0.3, 0.4, 0.5]]),
            np.array([[0.4, 0.5, 0.6]])
        ]

        # Create documentation
        doc = temp_dir / "product_info.txt"
        doc.write_text("""
        Product Information

        Our software comes in three editions:
        - Basic: $10/month, includes core features
        - Professional: $25/month, includes advanced analytics
        - Enterprise: $50/month, includes custom integrations

        All plans include 24/7 email support.
        Professional and Enterprise plans include phone support.
        """)

        # Setup chatbot
        processor = DocumentProcessor()
        processed_docs = processor.process_directory(temp_dir)

        vector_store = VectorStore()
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        vector_store.add_documents(all_chunks)

        chatbot = ChatBot()
        chatbot.vector_store = vector_store

        # Start conversation
        session = ChatSession(session_id="context_test", user_id="test_user")

        # Multi-turn conversation
        conversation = [
            "What pricing plans do you offer?",
            "What's included in the Professional plan?",
            "Does that plan include phone support?",
            "How much does it cost?"
        ]

        for question in conversation:
            # User message
            user_msg = chatbot.create_chat_message(
                content=question,
                role=MessageRole.USER
            )
            session.add_message(user_msg)

            # Bot response
            answer, found_relevant, citations = chatbot.answer_question(
                question, session=session
            )

            assistant_msg = chatbot.create_chat_message(
                content=answer,
                role=MessageRole.ASSISTANT,
                citations=citations
            )
            session.add_message(assistant_msg)

            # Verify response quality
            assert isinstance(answer, str)
            assert len(answer) > 0

        # Verify conversation context is maintained
        assert session.message_count == 8  # 4 questions + 4 responses
        context = session.get_conversation_context()
        assert len(context) > 0
        assert "pricing" in context.lower() or "plan" in context.lower()

    def test_error_recovery_workflow(self, temp_dir: Path):
        """Test error recovery and graceful degradation."""
        # Test with no documents (empty knowledge base)
        chatbot = ChatBot()

        # Should handle gracefully
        answer, found_relevant, citations = chatbot.answer_question(
            "How do I use your product?"
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert not found_relevant
        assert len(citations) == 0
        assert "documentation" in answer.lower() or "support" in answer.lower()

        # Test with corrupted vector store
        chatbot.vector_store = None

        answer, found_relevant, citations = chatbot.answer_question(
            "Another test question"
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert not found_relevant
        assert len(citations) == 0

