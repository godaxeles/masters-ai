"""Main chatbot service with RAG capabilities."""

import time
import uuid
from typing import List, Optional, Tuple

import structlog

from config import settings
from models.chat import ChatMessage, ChatSession, Citation, MessageRole, MessageType
from models.document import DocumentMetadata
from services.vector_store import VectorStore
from services.company_info import CompanyInfoService
from utils.metrics import QueryMetrics, metrics_collector
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ChatBot:
    """RAG-powered chatbot for customer support."""

    def __init__(self, vector_store_path: Optional[str] = None):
        """Initialize chatbot.

        Args:
            vector_store_path: Path to vector store directory
        """
        self.vector_store = VectorStore()
        self.company_info = CompanyInfoService()
        self.vector_store_path = vector_store_path or str(settings.vector_store_path)

        # Load vector store if it exists
        self._load_vector_store()

        logger.info("ChatBot initialized", vector_store_path=self.vector_store_path)

    def _load_vector_store(self) -> None:
        """Load vector store from disk if available."""
        try:
            from pathlib import Path
            store_path = Path(self.vector_store_path)

            if self.vector_store.exists(store_path):
                self.vector_store.load(store_path)
                logger.info("Vector store loaded successfully")
            else:
                logger.warning("Vector store not found, will need to process documents first")
        except Exception as e:
            logger.error("Failed to load vector store", error=str(e))

    def answer_question(
        self,
        question: str,
        session: Optional[ChatSession] = None,
        k: int = None,
        score_threshold: float = None
    ) -> Tuple[str, bool, List[Citation]]:
        """Answer a question using RAG.

        Args:
            question: User question
            session: Optional chat session for context
            k: Number of documents to retrieve
            score_threshold: Minimum similarity threshold

        Returns:
            Tuple of (answer, found_relevant_answer, citations)
        """
        start_time = time.time()

        k = k or settings.top_k_retrieval
        score_threshold = score_threshold or settings.similarity_threshold

        logger.info("Processing question", question=question[:100], k=k, threshold=score_threshold)

        # Start timing context
        with metrics_collector.timing_context("total_processing") as timing:
            try:
                # Retrieve relevant documents
                with metrics_collector.timing_context("retrieval") as retrieval_timing:
                    search_results = self.vector_store.search(
                        query=question,
                        k=k,
                        score_threshold=score_threshold
                    )

                # Generate response
                with metrics_collector.timing_context("generation") as generation_timing:
                    if search_results and search_results[0][1] >= score_threshold:
                        answer, citations = self._generate_answer_with_context(
                            question, search_results, session
                        )
                        found_relevant = True
                    else:
                        answer = self._generate_fallback_response(question, session)
                        citations = []
                        found_relevant = False

                # Record metrics
                query_metrics = QueryMetrics(
                    query=question,
                    timestamp=time.time(),
                    processing_time=timing.get("total_processing", 0),
                    retrieval_time=retrieval_timing.get("retrieval", 0),
                    generation_time=generation_timing.get("generation", 0),
                    retrieved_docs=len(search_results),
                    max_similarity_score=search_results[0][1] if search_results else 0.0,
                    response_length=len(answer),
                    citation_count=len(citations)
                )
                metrics_collector.record_query(query_metrics)

                logger.info(
                    "Question processed",
                    found_relevant=found_relevant,
                    citations_count=len(citations),
                    processing_time=timing.get("total_processing", 0)
                )

                return answer, found_relevant, citations

            except Exception as e:
                logger.error("Error processing question", error=str(e))
                metrics_collector.record_error("question_processing", str(e))
                return self._generate_error_response(), False, []

    def _generate_answer_with_context(
        self,
        question: str,
        search_results: List[Tuple[DocumentMetadata, float]],
        session: Optional[ChatSession] = None
    ) -> Tuple[str, List[Citation]]:
        """Generate answer using retrieved context.

        Args:
            question: User question
            search_results: Retrieved documents with similarity scores
            session: Optional chat session for context

        Returns:
            Tuple of (answer, citations)
        """
        # Extract the best context
        best_metadata, best_score = search_results[0]

        # Create citations
        citations = []
        context_parts = []

        for metadata, score in search_results[:3]:  # Use top 3 results
            citation = Citation(
                source=metadata.source,
                page=metadata.page,
                chunk_id=metadata.chunk_id,
                similarity_score=score,
                excerpt=metadata.content_preview
            )
            citations.append(citation)
            context_parts.append(metadata.content_preview)

        # Combine context
        context = "\n\n".join(context_parts)

        # Generate response based on context
        # This is a simplified response generation - in production you'd use an LLM
        answer = self._create_contextual_answer(question, context, session)

        return answer, citations

    def _create_contextual_answer(
        self,
        question: str,
        context: str,
        session: Optional[ChatSession] = None
    ) -> str:
        """Create an answer based on retrieved context.

        Args:
            question: User question
            context: Retrieved context
            session: Optional chat session

        Returns:
            Generated answer
        """
        # This is a simplified implementation
        # In production, you would use an LLM like OpenAI GPT or similar

        # Check if question is about company information
        if any(word in question.lower() for word in ['contact', 'phone', 'email', 'address', 'company']):
            company_info = self.company_info.get_company_info()
            return f"Here's our company information:\n\n{company_info}\n\nAdditionally, based on our documentation: {context[:300]}..."

        # Extract relevant information from context
        answer_parts = []

        # Add context-based answer
        if context:
            answer_parts.append(f"Based on our documentation: {context[:500]}...")

        # Add conversational context if available
        if session and session.message_count > 1:
            recent_context = session.get_conversation_context(max_tokens=500)
            if recent_context:
                answer_parts.append(f"\nConsidering our conversation history, this information should help address your question.")

        if not answer_parts:
            return "I found some relevant information in our documentation, but I'd like to provide you with more specific help. Could you please rephrase your question or provide more details?"

        return "\n\n".join(answer_parts)

    def _generate_fallback_response(
        self,
        question: str,
        session: Optional[ChatSession] = None
    ) -> str:
        """Generate fallback response when no relevant documents found.

        Args:
            question: User question
            session: Optional chat session

        Returns:
            Fallback response
        """
        # Check if it's a company info question
        if any(word in question.lower() for word in ['contact', 'phone', 'email', 'address']):
            return self.company_info.get_company_info()

        # Check if it's a greeting or general query
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(greeting in question.lower() for greeting in greetings):
            return f"Hello! I'm {settings.company_name}'s support assistant. I can help you find information from our documentation or create a support ticket if needed. What can I help you with today?"

        # Default fallback
        return (
            "I couldn't find specific information about that in our documentation. "
            "However, I'd be happy to create a support ticket for you so our team can "
            "provide personalized assistance. Would you like me to do that?"
        )

    def _generate_error_response(self) -> str:
        """Generate error response.

        Returns:
            Error response message
        """
        return (
            "I'm sorry, I encountered an error while processing your question. "
            "Please try again, or I can create a support ticket for you to get "
            "personalized help from our team."
        )

    def create_chat_message(
        self,
        content: str,
        role: MessageRole,
        message_type: MessageType = MessageType.TEXT,
        citations: Optional[List[Citation]] = None,
        processing_time: Optional[float] = None
    ) -> ChatMessage:
        """Create a chat message.

        Args:
            content: Message content
            role: Message role
            message_type: Type of message
            citations: Optional citations
            processing_time: Processing time in seconds

        Returns:
            ChatMessage object
        """
        return ChatMessage(
            message_id=str(uuid.uuid4()),
            role=role,
            content=content,
            message_type=message_type,
            citations=citations or [],
            processing_time=processing_time
        )

    def get_vector_store_stats(self) -> dict:
        """Get vector store statistics.

        Returns:
            Dictionary with vector store stats
        """
        return self.vector_store.get_stats()

    def is_ready(self) -> bool:
        """Check if chatbot is ready to answer questions.

        Returns:
            True if vector store is loaded and ready
        """
        return (
            self.vector_store.index is not None and
            len(self.vector_store.metadata) > 0
        )

