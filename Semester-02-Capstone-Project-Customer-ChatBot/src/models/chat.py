"""Chat and conversation data models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Type of message."""
    TEXT = "text"
    DOCUMENT_QUERY = "document_query"
    TICKET_REQUEST = "ticket_request"
    SYSTEM_INFO = "system_info"


class Citation(BaseModel):
    """Source citation for a response."""

    source: str = Field(..., description="Source document name")
    page: int = Field(default=0, description="Page number (0 for non-paginated)")
    chunk_id: str = Field(..., description="Chunk identifier")
    similarity_score: float = Field(..., description="Similarity score")
    excerpt: str = Field(..., description="Relevant text excerpt")

    def format_citation(self) -> str:
        """Format citation for display.

        Returns:
            Formatted citation string
        """
        if self.page > 0:
            return f"{self.source} (page {self.page})"
        else:
            return f"{self.source}"


class ChatMessage(BaseModel):
    """A single chat message."""

    message_id: str = Field(..., description="Unique message identifier")
    role: MessageRole = Field(..., description="Role of message sender")
    content: str = Field(..., description="Message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict = Field(default_factory=dict, description="Additional message metadata")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    processing_time: Optional[float] = Field(None, description="Time taken to process message")

    @property
    def has_citations(self) -> bool:
        """Check if message has citations."""
        return len(self.citations) > 0

    @property
    def formatted_content(self) -> str:
        """Get content with formatted citations.

        Returns:
            Content with citations appended
        """
        content = self.content
        if self.citations:
            citation_text = "\n\n**Sources:**\n" + "\n".join(
                f"- {citation.format_citation()}"
                for citation in self.citations
            )
            content += citation_text
        return content


class ConversationSummary(BaseModel):
    """Summary of conversation topics and context."""

    main_topics: List[str] = Field(default_factory=list, description="Main conversation topics")
    user_intent: Optional[str] = Field(None, description="Inferred user intent")
    unresolved_issues: List[str] = Field(default_factory=list, description="Unresolved user issues")
    relevant_documents: List[str] = Field(default_factory=list, description="Documents referenced")

    def add_topic(self, topic: str) -> None:
        """Add a topic to the conversation.

        Args:
            topic: Topic to add
        """
        if topic not in self.main_topics:
            self.main_topics.append(topic)

    def add_unresolved_issue(self, issue: str) -> None:
        """Add an unresolved issue.

        Args:
            issue: Issue to add
        """
        if issue not in self.unresolved_issues:
            self.unresolved_issues.append(issue)


class ChatSession(BaseModel):
    """A chat session with message history."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    messages: List[ChatMessage] = Field(default_factory=list, description="Session messages")
    summary: ConversationSummary = Field(default_factory=ConversationSummary, description="Conversation summary")
    is_active: bool = Field(default=True, description="Whether session is active")
    tickets_created: int = Field(default=0, description="Number of tickets created in session")

    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        """Get number of user messages."""
        return len([msg for msg in self.messages if msg.role == MessageRole.USER])

    @property
    def assistant_message_count(self) -> int:
        """Get number of assistant messages."""
        return len([msg for msg in self.messages if msg.role == MessageRole.ASSISTANT])

    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return (self.last_activity - self.started_at).total_seconds() / 60

    @property
    def avg_response_time(self) -> Optional[float]:
        """Get average response processing time."""
        processing_times = [
            msg.processing_time for msg in self.messages
            if msg.processing_time is not None and msg.role == MessageRole.ASSISTANT
        ]
        if processing_times:
            return sum(processing_times) / len(processing_times)
        return None

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.last_activity = datetime.now()

        # Update summary based on message
        if message.role == MessageRole.USER:
            # Extract topics from user message (simplified)
            words = message.content.lower().split()
            potential_topics = [word for word in words if len(word) > 5]
            for topic in potential_topics[:3]:  # Limit topics
                self.summary.add_topic(topic)

    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get recent messages for context.

        Args:
            count: Number of recent messages to return

        Returns:
            List of recent messages
        """
        return self.messages[-count:] if self.messages else []

    def get_conversation_context(self, max_tokens: int = 2000) -> str:
        """Get conversation context as a string.

        Args:
            max_tokens: Maximum tokens to include (approximate)

        Returns:
            Conversation context string
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars

        # Start from most recent messages and work backwards
        for message in reversed(self.messages):
            message_text = f"{message.role.value}: {message.content}\n"
            if total_chars + len(message_text) > max_chars:
                break
            context_parts.insert(0, message_text)
            total_chars += len(message_text)

        return "\n".join(context_parts)

    def end_session(self) -> None:
        """Mark session as ended."""
        self.is_active = False
        self.last_activity = datetime.now()


class UserFeedback(BaseModel):
    """User feedback on a chat interaction."""

    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    satisfied: bool = Field(..., description="Whether user was satisfied")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")

