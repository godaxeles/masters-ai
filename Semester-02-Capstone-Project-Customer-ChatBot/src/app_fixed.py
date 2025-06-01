"""Streamlit application for the Customer Support RAG Chatbot."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from .config import settings
from .models.chat import ChatSession, MessageRole, MessageType
from .models.ticket import TicketRequest, TicketPriority, TicketCategory
from .services.chatbot import ChatBot
from .services.ticket_service import TicketService
from .utils.logging_config import setup_logging, get_logger
from .utils.metrics import metrics_collector

# Setup logging
setup_logging(log_file=settings.logs_path / "app.log")
logger = get_logger(__name__)


class ChatApp:
    """Main chat application class."""

    def __init__(self):
        """Initialize the chat application."""
        self.chatbot = ChatBot()
        self.ticket_service = TicketService()

        # Initialize session state
        if 'session' not in st.session_state:
            st.session_state.session = ChatSession(
                session_id=str(uuid.uuid4()),
                user_id="streamlit_user"
            )

        if 'show_ticket_form' not in st.session_state:
            st.session_state.show_ticket_form = False

        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""

        logger.info("Chat application initialized")
