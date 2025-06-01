"""Support ticket data models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TicketPriority(str, Enum):
    """Priority levels for support tickets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    """Status of support tickets."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketCategory(str, Enum):
    """Categories for support tickets."""
    GENERAL_INQUIRY = "general_inquiry"
    TECHNICAL_ISSUE = "technical_issue"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    DOCUMENTATION = "documentation"
    ACCOUNT_ISSUE = "account_issue"
    OTHER = "other"


class SupportTicket(BaseModel):
    """A customer support ticket."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    title: str = Field(..., description="Ticket title/summary")
    description: str = Field(..., description="Detailed ticket description")
    user_name: str = Field(..., description="Name of user who created ticket")
    user_email: str = Field(..., description="Email of user who created ticket")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Ticket priority")
    category: TicketCategory = Field(default=TicketCategory.GENERAL_INQUIRY, description="Ticket category")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Ticket status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    labels: List[str] = Field(default_factory=list, description="Ticket labels")
    original_query: Optional[str] = Field(None, description="Original user query that led to ticket")
    chat_context: Optional[str] = Field(None, description="Chat conversation context")
    external_id: Optional[str] = Field(None, description="External system ticket ID (e.g., GitHub issue number)")
    external_url: Optional[str] = Field(None, description="External system ticket URL")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

    @property
    def age_hours(self) -> float:
        """Get ticket age in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    @property
    def age_days(self) -> float:
        """Get ticket age in days."""
        return self.age_hours / 24

    def add_label(self, label: str) -> None:
        """Add a label to the ticket.

        Args:
            label: Label to add
        """
        if label not in self.labels:
            self.labels.append(label)
            self.updated_at = datetime.now()

    def remove_label(self, label: str) -> None:
        """Remove a label from the ticket.

        Args:
            label: Label to remove
        """
        if label in self.labels:
            self.labels.remove(label)
            self.updated_at = datetime.now()

    def update_status(self, status: TicketStatus) -> None:
        """Update ticket status.

        Args:
            status: New status
        """
        self.status = status
        self.updated_at = datetime.now()

    def set_external_reference(self, external_id: str, external_url: str) -> None:
        """Set external system reference.

        Args:
            external_id: External system ID
            external_url: External system URL
        """
        self.external_id = external_id
        self.external_url = external_url
        self.updated_at = datetime.now()


class TicketRequest(BaseModel):
    """Request to create a new support ticket."""

    title: str = Field(..., description="Ticket title", min_length=5, max_length=200)
    description: str = Field(..., description="Ticket description", min_length=10)
    user_name: str = Field(..., description="User name", min_length=2, max_length=100)
    user_email: str = Field(..., description="User email")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Ticket priority")
    category: TicketCategory = Field(default=TicketCategory.GENERAL_INQUIRY, description="Ticket category")
    labels: List[str] = Field(default_factory=list, description="Initial labels")
    original_query: Optional[str] = Field(None, description="Original query")
    chat_context: Optional[str] = Field(None, description="Chat context")

    def to_ticket(self, ticket_id: str) -> SupportTicket:
        """Convert request to a full ticket.

        Args:
            ticket_id: Unique ticket identifier

        Returns:
            SupportTicket instance
        """
        return SupportTicket(
            ticket_id=ticket_id,
            title=self.title,
            description=self.description,
            user_name=self.user_name,
            user_email=self.user_email,
            priority=self.priority,
            category=self.category,
            labels=self.labels.copy(),
            original_query=self.original_query,
            chat_context=self.chat_context
        )


class TicketMetrics(BaseModel):
    """Metrics for ticket management."""

    total_tickets: int = Field(default=0, description="Total tickets created")
    open_tickets: int = Field(default=0, description="Currently open tickets")
    resolved_tickets: int = Field(default=0, description="Resolved tickets")
    avg_resolution_time_hours: Optional[float] = Field(None, description="Average resolution time")
    tickets_by_category: Dict[str, int] = Field(default_factory=dict, description="Tickets by category")
    tickets_by_priority: Dict[str, int] = Field(default_factory=dict, description="Tickets by priority")
    satisfaction_score: Optional[float] = Field(None, description="Average satisfaction score")

    def update_metrics(self, tickets: List[SupportTicket]) -> None:
        """Update metrics based on ticket list.

        Args:
            tickets: List of tickets to analyze
        """
        self.total_tickets = len(tickets)
        self.open_tickets = len([t for t in tickets if t.status in [TicketStatus.OPEN, TicketStatus.IN_PROGRESS]])
        self.resolved_tickets = len([t for t in tickets if t.status == TicketStatus.RESOLVED])

        # Calculate average resolution time for resolved tickets
        resolved_tickets = [t for t in tickets if t.status == TicketStatus.RESOLVED]
        if resolved_tickets:
            total_resolution_time = sum(t.age_hours for t in resolved_tickets)
            self.avg_resolution_time_hours = total_resolution_time / len(resolved_tickets)

        # Count by category
        self.tickets_by_category = {}
        for ticket in tickets:
            category = ticket.category.value
            self.tickets_by_category[category] = self.tickets_by_category.get(category, 0) + 1

        # Count by priority
        self.tickets_by_priority = {}
        for ticket in tickets:
            priority = ticket.priority.value
            self.tickets_by_priority[priority] = self.tickets_by_priority.get(priority, 0) + 1

