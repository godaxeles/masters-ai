"""Unit tests for chatbot service."""

import pytest
from unittest.mock import patch, MagicMock

from src.services.chatbot import ChatBot
from src.models.chat import ChatMessage, ChatSession, MessageRole, Citation
from src.models.document import DocumentMetadata


@pytest.mark.unit
@pytest.mark.rag
class TestChatBot:
    """Test chatbot functionality."""

    def test_init(self):
        """Test chatbot initialization."""
        with patch.object(ChatBot, '_load_vector_store'):
            chatbot = ChatBot()
            assert chatbot.vector_store is not None
            assert chatbot.company_info is not None

    def test_is_ready_empty_store(self):
        """Test is_ready with empty vector store."""
        with patch.object(ChatBot, '_load_vector_store'):
            chatbot = ChatBot()
            assert not chatbot.is_ready()

    def test_is_ready_populated_store(self, chatbot):
        """Test is_ready with populated vector store."""
        assert chatbot.is_ready()

    def test_answer_question_with_relevant_results(self, chatbot):
        """Test answering question with relevant search results."""
        answer, found_relevant, citations = chatbot.answer_question("test content")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(found_relevant, bool)
        assert isinstance(citations, list)

        if found_relevant:
            assert len(citations) > 0
            assert all(isinstance(citation, Citation) for citation in citations)

    def test_answer_question_no_relevant_results(self, chatbot):
        """Test answering question with no relevant results."""
        # Query that should not match anything
        answer, found_relevant, citations = chatbot.answer_question("xyz123 nonexistent query")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert not found_relevant
        assert len(citations) == 0

    def test_answer_company_info_question(self, chatbot):
        """Test answering company information questions."""
        questions = ["contact information", "phone number", "email address", "company details"]

        for question in questions:
            answer, found_relevant, citations = chatbot.answer_question(question)
            assert isinstance(answer, str)
            assert len(answer) > 0
            # Company info questions might be handled even without documents

    def test_answer_greeting(self, chatbot):
        """Test answering greetings."""
        greetings = ["hello", "hi", "good morning", "hey there"]

        for greeting in greetings:
            answer, found_relevant, citations = chatbot.answer_question(greeting)
            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_create_chat_message(self, chatbot):
        """Test creating chat messages."""
        message = chatbot.create_chat_message(
            content="Test message",
            role=MessageRole.USER
        )

        assert isinstance(message, ChatMessage)
        assert message.content == "Test message"
        assert message.role == MessageRole.USER
        assert len(message.message_id) > 0

    def test_create_chat_message_with_citations(self, chatbot):
        """Test creating chat messages with citations."""
        citations = [
            Citation(
                source="test.txt",
                page=1,
                chunk_id="chunk_1",
                similarity_score=0.8,
                excerpt="Test excerpt"
            )
        ]

        message = chatbot.create_chat_message(
            content="Test message",
            role=MessageRole.ASSISTANT,
            citations=citations
        )

        assert len(message.citations) == 1
        assert message.has_citations
        assert "**Sources:**" in message.formatted_content

    def test_get_vector_store_stats(self, chatbot):
        """Test getting vector store statistics."""
        stats = chatbot.get_vector_store_stats()

        assert isinstance(stats, dict)
        assert "total_vectors" in stats
        assert "unique_sources" in stats

    def test_generate_fallback_response(self, chatbot):
        """Test fallback response generation."""
        response = chatbot._generate_fallback_response("unknown query")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "documentation" in response.lower() or "support" in response.lower()

    def test_generate_error_response(self, chatbot):
        """Test error response generation."""
        response = chatbot._generate_error_response()

        assert isinstance(response, str)
        assert len(response) > 0
        assert "error" in response.lower() or "sorry" in response.lower()

    @patch('src.services.chatbot.metrics_collector')
    def test_answer_question_metrics(self, mock_metrics, chatbot):
        """Test that metrics are recorded for questions."""
        chatbot.answer_question("test query")

        # Verify metrics were recorded
        mock_metrics.record_query.assert_called_once()
        call_args = mock_metrics.record_query.call_args[0][0]
        assert call_args.query == "test query"
        assert call_args.processing_time > 0

    def test_answer_question_with_session_context(self, chatbot, chat_session):
        """Test answering questions with session context."""
        # Add some messages to session for context
        chat_session.add_message(ChatMessage(
            message_id="msg1",
            role=MessageRole.USER,
            content="Previous question about testing"
        ))

        answer, found_relevant, citations = chatbot.answer_question(
            "follow up question",
            session=chat_session
        )

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch('src.services.chatbot.logger')
    def test_error_handling(self, mock_logger, chatbot):
        """Test error handling in answer_question."""
        # Mock vector store to raise exception
        chatbot.vector_store.search = MagicMock(side_effect=Exception("Test error"))

        answer, found_relevant, citations = chatbot.answer_question("test query")

        assert not found_relevant
        assert len(citations) == 0
        assert "error" in answer.lower() or "sorry" in answer.lower()
        mock_logger.error.assert_called()

