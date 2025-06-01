"""Integration tests for external API interactions."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.services.ticket_service import TicketService
from src.models.ticket import TicketRequest


@pytest.mark.integration
class TestAPIIntegration:
    """Test external API integrations."""

    @patch('requests.post')
    @patch('requests.get')
    def test_github_api_integration(self, mock_get, mock_post):
        """Test GitHub API integration for ticket creation."""
        # Mock connection test
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock ticket creation
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {
            'number': 456,
            'html_url': 'https://github.com/test/repo/issues/456',
            'id': 123456,
            'state': 'open'
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        service = TicketService()

        # Test connection
        assert service.test_connection() is True

        # Test ticket creation
        ticket_request = TicketRequest(
            title="Integration Test Ticket",
            description="This is a test ticket created during integration testing.",
            user_name="Integration Tester",
            user_email="tester@example.com"
        )

        ticket = service.create_ticket(ticket_request)

        # Verify ticket was created correctly
        assert ticket.external_id == "456"
        assert ticket.external_url == 'https://github.com/test/repo/issues/456'

        # Verify API calls
        mock_get.assert_called_once()
        mock_post.assert_called_once()

        # Check POST request payload
        post_call_args = mock_post.call_args
        payload = post_call_args[1]['json']

        assert payload['title'] == "Integration Test Ticket"
        assert "This is a test ticket" in payload['body']
        assert "Integration Tester" in payload['body']
        assert "tester@example.com" in payload['body']
        assert 'support' in payload['labels']

    @patch('requests.post')
    def test_github_api_error_handling(self, mock_post):
        """Test GitHub API error handling."""
        # Test different types of API errors
        error_scenarios = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.HTTPError("HTTP 401 Unauthorized"),
            requests.exceptions.RequestException("General request error")
        ]

        service = TicketService()

        for error in error_scenarios:
            mock_post.side_effect = error

            ticket_request = TicketRequest(
                title="Test Error Handling",
                description="Testing error scenarios",
                user_name="Test User",
                user_email="test@example.com"
            )

            with pytest.raises(Exception) as exc_info:
                service.create_ticket(ticket_request)

            assert "Failed to create support ticket" in str(exc_info.value)

    @patch('requests.post')
    def test_github_api_rate_limiting(self, mock_post):
        """Test GitHub API rate limiting handling."""
        # Mock rate limit response
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            'message': 'API rate limit exceeded',
            'documentation_url': 'https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting'
        }

        error = requests.exceptions.HTTPError()
        error.response = mock_response
        mock_post.side_effect = error

        service = TicketService()

        ticket_request = TicketRequest(
            title="Rate Limit Test",
            description="Testing rate limit handling",
            user_name="Test User",
            user_email="test@example.com"
        )

        with pytest.raises(Exception) as exc_info:
            service.create_ticket(ticket_request)

        assert "Failed to create support ticket" in str(exc_info.value)

    @patch('requests.get')
    def test_github_repository_validation(self, mock_get):
        """Test GitHub repository access validation."""
        scenarios = [
            # Valid repository
            (200, {'name': 'test-repo', 'full_name': 'user/test-repo'}, True),
            # Repository not found
            (404, {'message': 'Not Found'}, False),
            # No access
            (403, {'message': 'Forbidden'}, False),
            # Invalid token
            (401, {'message': 'Bad credentials'}, False)
        ]

        service = TicketService()

        for status_code, response_data, expected_result in scenarios:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.json.return_value = response_data

            if status_code == 200:
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
            else:
                error = requests.exceptions.HTTPError()
                error.response = mock_response
                mock_get.side_effect = error

            result = service.test_connection()
            assert result == expected_result

    def test_api_timeout_configuration(self):
        """Test that API calls have proper timeout configuration."""
        service = TicketService()

        # Check that timeout is configured for requests
        # This is more of a code inspection test
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:

            # Test connection call
            try:
                service.test_connection()
            except:
                pass  # We just want to check the call parameters

            if mock_get.called:
                call_kwargs = mock_get.call_args[1]
                assert 'timeout' in call_kwargs
                assert call_kwargs['timeout'] > 0

            # Test ticket creation call
            mock_post_response = MagicMock()
            mock_post_response.json.return_value = {'number': 1, 'html_url': 'http://test.com'}
            mock_post_response.raise_for_status.return_value = None
            mock_post.return_value = mock_post_response

            ticket_request = TicketRequest(
                title="Timeout Test",
                description="Testing timeout configuration",
                user_name="Test User",
                user_email="test@example.com"
            )

            try:
                service.create_ticket(ticket_request)
            except:
                pass

            if mock_post.called:
                call_kwargs = mock_post.call_args[1]
                assert 'timeout' in call_kwargs
                assert call_kwargs['timeout'] > 0

