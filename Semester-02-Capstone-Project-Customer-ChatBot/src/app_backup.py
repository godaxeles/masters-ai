"""Streamlit application for the Customer Support RAG Chatbot."""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Fix imports for Streamlit direct execution
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import settings
from models.chat import ChatSession, MessageRole, MessageType
from models.ticket import TicketRequest, TicketPriority, TicketCategory
from services.chatbot import ChatBot
from services.ticket_service import TicketService
from utils.logging_config import setup_logging, get_logger
from utils.metrics import metrics_collector

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

    def run_streamlit(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title=f"{settings.company_name} - Support Chat",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stButton > button {
            width: 100%;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Sidebar
        self._render_sidebar()

        # Main content
        st.title(f"🤖 {settings.company_name} Support Assistant")
        st.markdown("Ask me anything about our products and services. I can help you find information or create a support ticket.")

        # Check if chatbot is ready
        if not self.chatbot.is_ready():
            st.error("⚠️ The chatbot is not ready. Please run document ingestion first: `python -m src.ingest`")
            return

        # Chat interface
        self._render_chat_interface()

        # Ticket form
        if st.session_state.show_ticket_form:
            self._render_ticket_form()

    def _render_sidebar(self):
        """Render the sidebar with information and controls."""
        with st.sidebar:
            st.header("📊 Information")

            # System status
            with st.expander("🔧 System Status", expanded=True):
                if self.chatbot.is_ready():
                    st.success("✅ Chatbot Ready")
                    stats = self.chatbot.get_vector_store_stats()
                    st.metric("Documents", stats.get('unique_sources', 0))
                    st.metric("Knowledge Chunks", stats.get('total_vectors', 0))
                else:
                    st.error("❌ Chatbot Not Ready")

                # Test GitHub connection
                if self.ticket_service.test_connection():
                    st.success("✅ GitHub Connected")
                else:
                    st.error("❌ GitHub Not Connected")

            # Metrics
            with st.expander("📈 Session Metrics"):
                session = st.session_state.session
                st.metric("Messages", session.message_count)
                st.metric("Duration (min)", round(session.duration_minutes, 1))

                if session.tickets_created > 0:
                    st.metric("Tickets Created", session.tickets_created)

            # Company information
            with st.expander("🏢 Company Info"):
                st.markdown(self.chatbot.company_info.get_company_info())

            # Actions
            st.header("🔧 Actions")

            if st.button("🎫 Create Support Ticket"):
                st.session_state.show_ticket_form = True
                st.rerun()

            if st.button("🔄 New Chat Session"):
                st.session_state.session = ChatSession(
                    session_id=str(uuid.uuid4()),
                    user_id="streamlit_user"
                )
                st.session_state.show_ticket_form = False
                st.rerun()

    def _render_chat_interface(self):
        """Render the main chat interface."""
        session = st.session_state.session

        # Display chat history
        for message in session.messages:
            with st.chat_message(message.role.value):
                st.markdown(message.formatted_content)

        # Chat input
        if prompt := st.chat_input("Ask me anything about our products and services..."):
            self._handle_user_message(prompt)

    def _handle_user_message(self, prompt: str):
        """Handle a user message."""
        session = st.session_state.session
        st.session_state.last_query = prompt

        # Add user message
        user_message = self.chatbot.create_chat_message(
            content=prompt,
            role=MessageRole.USER,
            message_type=MessageType.DOCUMENT_QUERY
        )
        session.add_message(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, found_relevant, citations = self.chatbot.answer_question(
                    question=prompt,
                    session=session
                )

            # Create assistant message
            assistant_message = self.chatbot.create_chat_message(
                content=answer,
                role=MessageRole.ASSISTANT,
                message_type=MessageType.DOCUMENT_QUERY,
                citations=citations
            )
            session.add_message(assistant_message)

            # Display response
            st.markdown(assistant_message.formatted_content)

            # Show action buttons if no relevant answer found
            if not found_relevant:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎫 Create Support Ticket", key=f"ticket_{user_message.message_id}"):
                        st.session_state.show_ticket_form = True
                        st.rerun()

        # Auto-scroll to bottom
        st.rerun()

    def _render_ticket_form(self):
        """Render the support ticket creation form."""
        st.header("🎫 Create Support Ticket")

        with st.form("ticket_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                user_name = st.text_input(
                    "Your Name *",
                    placeholder="Enter your full name"
                )
                user_email = st.text_input(
                    "Email Address *",
                    placeholder="your.email@example.com"
                )

            with col2:
                priority = st.selectbox(
                    "Priority",
                    options=[p.value for p in TicketPriority],
                    index=1  # Medium
                )
                category = st.selectbox(
                    "Category",
                    options=[c.value.replace('_', ' ').title() for c in TicketCategory],
                    index=0  # General Inquiry
                )

            title = st.text_input(
                "Issue Summary *",
                placeholder="Brief description of your issue",
                value=f"Support needed: {st.session_state.last_query[:50]}..." if st.session_state.last_query else ""
            )

            description = st.text_area(
                "Detailed Description *",
                placeholder="Please provide a detailed description of your issue or question...",
                height=150,
                value=st.session_state.last_query if st.session_state.last_query else ""
            )

            include_chat = st.checkbox(
                "Include chat conversation context",
                value=True,
                help="Include your recent conversation with the chatbot for context"
            )

            col1, col2 = st.columns(2)

            with col1:
                submitted = st.form_submit_button("🎫 Create Ticket", type="primary")

            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.show_ticket_form = False
                    st.rerun()

            if submitted:
                self._create_ticket(
                    title=title,
                    description=description,
                    user_name=user_name,
                    user_email=user_email,
                    priority=TicketPriority(priority),
                    category=TicketCategory(category.lower().replace(' ', '_')),
                    include_chat=include_chat
                )

    def _create_ticket(self, title: str, description: str, user_name: str, 
                      user_email: str, priority: TicketPriority, 
                      category: TicketCategory, include_chat: bool):
        """Create a support ticket."""
        # Validate required fields
        if not all([title, description, user_name, user_email]):
            st.error("Please fill in all required fields marked with *")
            return

        try:
            # Prepare chat context if requested
            chat_context = None
            if include_chat:
                session = st.session_state.session
                chat_context = session.get_conversation_context(max_tokens=2000)

            # Create ticket request
            ticket_request = TicketRequest(
                title=title,
                description=description,
                user_name=user_name,
                user_email=user_email,
                priority=priority,
                category=category,
                original_query=st.session_state.last_query,
                chat_context=chat_context
            )

            # Create ticket
            with st.spinner("Creating support ticket..."):
                ticket = self.ticket_service.create_ticket(ticket_request)

            # Update session
            session = st.session_state.session
            session.tickets_created += 1

            # Show success message
            st.success(f"✅ Support ticket created successfully!")
            st.info(f"🎫 Ticket ID: {ticket.ticket_id}")
            if ticket.external_url:
                st.info(f"🔗 Track your ticket: {ticket.external_url}")

            # Reset form
            st.session_state.show_ticket_form = False
            st.session_state.last_query = ""

            # Add system message to chat
            system_message = self.chatbot.create_chat_message(
                content=f"✅ Support ticket created successfully! Ticket ID: {ticket.ticket_id}",
                role=MessageRole.ASSISTANT,
                message_type=MessageType.SYSTEM_INFO
            )
            session.add_message(system_message)

            st.rerun()

        except Exception as e:
            logger.error("Failed to create ticket", error=str(e))
            st.error(f"❌ Failed to create ticket: {str(e)}")


def run_gradio_app():
    """Run the Gradio version of the application."""
    import gradio as gr
    
    chatbot = ChatBot()
    ticket_service = TicketService()

    def respond(message, history):
        """Handle chat response in Gradio."""
        if not chatbot.is_ready():
            return "⚠️ The chatbot is not ready. Please run document ingestion first."

        answer, found_relevant, citations = chatbot.answer_question(message)

        if not found_relevant:
            answer += "\n\n💡 If this doesn't answer your question, you can create a support ticket for personalized help."

        return answer

    # Create Gradio interface
    demo = gr.ChatInterface(
        respond,
        title=f"🤖 {settings.company_name} Support Assistant",
        description="Ask me anything about our products and services. I can help you find information from our documentation.",
        theme="soft",
        examples=[
            "What are your contact details?",
            "How can I get support?",
            "Tell me about your products",
        ]
    )

    return demo


def main():
    """Main application entry point."""
    if settings.use_gradio:
        logger.info("Starting Gradio application")
        app = run_gradio_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    else:
        logger.info("Starting Streamlit application")
        app = ChatApp()
        app.run_streamlit()


if __name__ == "__main__":
    main()
