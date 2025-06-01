"""Mock ticket service for demonstration purposes."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import structlog

from config import settings
from models.ticket import TicketRequest, TicketResponse, TicketStatus

logger = structlog.get_logger(__name__)


class MockTicketService:
    """Mock ticket service that saves tickets to local files for demo."""

    def __init__(self):
        """Initialize mock ticket service."""
        self.tickets_dir = Path("data/tickets")
        self.tickets_dir.mkdir(exist_ok=True)
        logger.info("Mock ticket service initialized", tickets_dir=str(self.tickets_dir))

    def test_connection(self) -> bool:
        """Test connection - always returns True for mock."""
        return True

    def create_ticket(self, ticket_request: TicketRequest) -> TicketResponse:
        """Create a mock support ticket saved to local file.

        Args:
            ticket_request: Ticket creation request

        Returns:
            TicketResponse with mock ticket details
        """
        # Generate ticket
        ticket_id = f"MOCK-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()
        
        # Create ticket data
        ticket_data = {
            "ticket_id": ticket_id,
            "title": ticket_request.title,
            "description": ticket_request.description,
            "user_name": ticket_request.user_name,
            "user_email": ticket_request.user_email,
            "priority": ticket_request.priority.value,
            "category": ticket_request.category.value,
            "status": "open",
            "created_at": timestamp,
            "original_query": ticket_request.original_query,
            "chat_context": ticket_request.chat_context
        }
        
        # Save to file
        ticket_file = self.tickets_dir / f"{ticket_id}.json"
        with open(ticket_file, 'w') as f:
            json.dump(ticket_data, f, indent=2)
        
        logger.info("Mock ticket created", ticket_id=ticket_id, file=str(ticket_file))
        
        # Return response
        return TicketResponse(
            ticket_id=ticket_id,
            status=TicketStatus.OPEN,
            external_url=f"file://{ticket_file.absolute()}",
            created_at=datetime.fromisoformat(timestamp)
        )

    def get_tickets_summary(self) -> Dict[str, Any]:
        """Get summary of all mock tickets."""
        ticket_files = list(self.tickets_dir.glob("*.json"))
        
        return {
            "total_tickets": len(ticket_files),
            "tickets_directory": str(self.tickets_dir.absolute()),
            "recent_tickets": [f.stem for f in sorted(ticket_files)[-5:]]
        }
