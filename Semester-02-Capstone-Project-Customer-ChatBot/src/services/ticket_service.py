"""Support ticket service for creating and managing tickets."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import requests
import structlog

from config import settings
from models.ticket import SupportTicket, TicketRequest, TicketPriority, TicketCategory
from utils.logging_config import get_logger
from utils.validators import validate_github_token, validate_github_repo

logger = get_logger(__name__)


class TicketService:
    """Service for creating and managing support tickets."""

    def __init__(self):
        """Initialize ticket service."""
        self.github_token = settings.github_token
        self.github_repo = settings.github_repo
        self.base_url = "https://api.github.com"

        # Validate configuration
        if not validate_github_token(self.github_token):
            logger.error("Invalid GitHub token format")

        if not validate_github_repo(self.github_repo):
            logger.error("Invalid GitHub repository format")

        logger.info("Ticket service initialized", repo=self.github_repo)

    def create_ticket(self, ticket_request: TicketRequest) -> SupportTicket:
        """Create a new support ticket.

        Args:
            ticket_request: Ticket creation request

        Returns:
            Created support ticket

        Raises:
            Exception: If ticket creation fails
        """
        # Generate unique ticket ID
        ticket_id = str(uuid.uuid4())[:8]

        # Create ticket object
        ticket = ticket_request.to_ticket(ticket_id)

        logger.info(
            "Creating support ticket",
            ticket_id=ticket_id,
            title=ticket.title,
            user_email=ticket.user_email
        )

        try:
            # Create GitHub issue
            github_issue = self._create_github_issue(ticket)

            # Update ticket with GitHub information
            ticket.set_external_reference(
                external_id=str(github_issue['number']),
                external_url=github_issue['html_url']
            )

            logger.info(
                "Support ticket created successfully",
                ticket_id=ticket_id,
                github_issue=github_issue['number'],
                github_url=github_issue['html_url']
            )

            return ticket

        except Exception as e:
            logger.error(
                "Failed to create support ticket",
                ticket_id=ticket_id,
                error=str(e)
            )
            raise Exception(f"Failed to create support ticket: {str(e)}")

    def _create_github_issue(self, ticket: SupportTicket) -> Dict:
        """Create a GitHub issue for the ticket.

        Args:
            ticket: Support ticket to create issue for

        Returns:
            GitHub issue data

        Raises:
            Exception: If GitHub API call fails
        """
        url = f"{self.base_url}/repos/{self.github_repo}/issues"

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }

        # Create issue body
        body_parts = [
            f"**Support Ticket ID:** {ticket.ticket_id}",
            f"**User:** {ticket.user_name} ({ticket.user_email})",
            f"**Priority:** {ticket.priority.value.title()}",
            f"**Category:** {ticket.category.value.replace('_', ' ').title()}",
            f"**Created:** {ticket.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Description",
            ticket.description
        ]

        if ticket.original_query:
            body_parts.extend([
                "",
                "## Original Query",
                f"```\n{ticket.original_query}\n```"
            ])

        if ticket.chat_context:
            body_parts.extend([
                "",
                "## Chat Context",
                f"```\n{ticket.chat_context}\n```"
            ])

        # Prepare labels
        labels = ["support", ticket.priority.value, ticket.category.value]
        if ticket.labels:
            labels.extend(ticket.labels)

        payload = {
            "title": ticket.title,
            "body": "\n".join(body_parts),
            "labels": labels
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            issue_data = response.json()

            logger.info(
                "GitHub issue created",
                issue_number=issue_data['number'],
                issue_url=issue_data['html_url']
            )

            return issue_data

        except requests.exceptions.RequestException as e:
            logger.error(
                "GitHub API request failed",
                url=url,
                error=str(e),
                status_code=getattr(e.response, 'status_code', None)
            )
            raise Exception(f"GitHub API error: {str(e)}")

    def test_connection(self) -> bool:
        """Test connection to GitHub API.

        Returns:
            True if connection successful
        """
        url = f"{self.base_url}/repos/{self.github_repo}"

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            logger.info("GitHub connection test successful")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(
                "GitHub connection test failed",
                error=str(e),
                status_code=getattr(e.response, 'status_code', None)
            )
            return False

    def create_quick_ticket(
        self,
        title: str,
        description: str,
        user_name: str,
        user_email: str,
        original_query: Optional[str] = None,
        chat_context: Optional[str] = None,
        priority: TicketPriority = TicketPriority.MEDIUM,
        category: TicketCategory = TicketCategory.GENERAL_INQUIRY
    ) -> SupportTicket:
        """Create a ticket with minimal information.

        Args:
            title: Ticket title
            description: Ticket description
            user_name: User name
            user_email: User email
            original_query: Original user query
            chat_context: Chat conversation context
            priority: Ticket priority
            category: Ticket category

        Returns:
            Created support ticket
        """
        ticket_request = TicketRequest(
            title=title,
            description=description,
            user_name=user_name,
            user_email=user_email,
            priority=priority,
            category=category,
            original_query=original_query,
            chat_context=chat_context
        )

        return self.create_ticket(ticket_request)

    def get_ticket_labels(self) -> List[str]:
        """Get available ticket labels.

        Returns:
            List of available labels
        """
        return [
            "support",
            "bug",
            "enhancement",
            "documentation",
            "question",
            "urgent",
            "high-priority",
            "low-priority"
        ]

