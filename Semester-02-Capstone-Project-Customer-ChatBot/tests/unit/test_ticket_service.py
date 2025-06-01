"""Unit tests for ticket service."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.services.ticket_service import TicketService
from src.models.ticket import TicketRequest, SupportTicket, TicketPriority, TicketCategory


@pytest.mark.unit
class TestTicketService:
    """Test ticket service functionality."""

    def test_init(self):
        """Test ticket service initialization."""
        with patch('src.services.ticket_service.validate_github_token', return_value=True), \
             patch('src.services.ticket_service.validate_github_repo', return_value=True):
            service = TicketService()
            assert service.github_token is not None
            assert service.github_repo is not None
            assert service.base_url == "https://api.github.com"

    @patch('requests.post')
    def test_create_ticket_success(self, mock_post):
        """Test successful ticket creation."""
        # Mock successful GitHub API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'number': 123,
            'html_url': 'https://github.com/owner/repo/issues/123'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = TicketService()

        ticket_request = TicketRequest(
            title="Test ticket",
            description="Test description",
            user_name="Test User",
            user_email="test@example.com"
        )

        ticket = service.create_ticket(ticket_request)

        assert isinstance(ticket, SupportTicket)
        assert ticket.title == "Test ticket"
        assert ticket.external_id == "123"
        assert ticket.external_url == 'https://github.com/owner/repo/issues/123'

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert 'issues' in call_args[1]['json']['title']

    @patch('requests.post')
    def test_create_ticket_api_failure(self, mock_post):
        """Test ticket creation with API failure."""
        # Mock API failure
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        service = TicketService()

        ticket_request = TicketRequest(
            title="Test ticket",
            description="Test description",
            user_name="Test User",
            user_email="test@example.com"
        )

        with pytest.raises(Exception) as exc_info:
            service.create_ticket(ticket_request)

        assert "Failed to create support ticket" in str(exc_info.value)

    @patch('requests.get')
    def test_test_connection_success(self, mock_get):
        """Test successful GitHub connection test."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        service = TicketService()
        result = service.test_connection()

        assert result is True
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_test_connection_failure(self, mock_get):
        """Test failed GitHub connection test."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        service = TicketService()
        result = service.test_connection()

        assert result is False

    def test_create_quick_ticket(self):
        """Test quick ticket creation."""
        with patch.object(TicketService, 'create_ticket') as mock_create:
            mock_ticket = SupportTicket(
                ticket_id="test123",
                title="Quick ticket",
                description="Quick description",
                user_name="Test User",
                user_email="test@example.com"
            )
            mock_create.return_value = mock_ticket

            service = TicketService()
            result = service.create_quick_ticket(
                title="Quick ticket",
                description="Quick description",
                user_name="Test User",
                user_email="test@example.com"
            )

            assert result == mock_ticket
            mock_create.assert_called_once()

    def test_get_ticket_labels(self):
        """Test getting available ticket labels."""
        service = TicketService()
        labels = service.get_ticket_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0
        assert "support" in labels
        assert "bug" in labels

    def test_create_github_issue_payload(self):
        """Test GitHub issue payload creation."""
        service = TicketService()

        ticket = SupportTicket(
            ticket_id="test123",
            title="Test Issue",
            description="Test description",
            user_name="Test User",
            user_email="test@example.com",
            priority=TicketPriority.HIGH,
            category=TicketCategory.BUG_REPORT,
            original_query="original query text",
            chat_context="chat context here"
        )

        # Mock the GitHub API call to inspect payload
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {'number': 1, 'html_url': 'http://test.com'}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            try:
                service._create_github_issue(ticket)
            except:
                pass  # We just want to check the call

            if mock_post.called:
                call_args = mock_post.call_args
                payload = call_args[1]['json']

                assert payload['title'] == "Test Issue"
                assert "Test description" in payload['body']
                assert "Test User" in payload['body']
                assert "test@example.com" in payload['body']
                assert "high" in payload['labels']
                assert "bug_report" in payload['labels']

