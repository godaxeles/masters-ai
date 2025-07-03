#!/usr/bin/env python3
"""
RAG Customer Support Chatbot for HuggingFace Spaces Deployment
Simplified version based on the full implementation
"""

import streamlit as st
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
import json

# Simplified imports for HuggingFace deployment
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

# Configuration
COMPANY_NAME = os.getenv("COMPANY_NAME", "TechCorp")
COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "support@techcorp.com")
COMPANY_PHONE = os.getenv("COMPANY_PHONE", "+1-555-0123")
COMPANY_WEBSITE = os.getenv("COMPANY_WEBSITE", "https://techcorp.com")

class SimpleRAGChatbot:
    """Simplified RAG chatbot for demonstration purposes."""
    
    def __init__(self):
        """Initialize the chatbot with sample data."""
        self.embedder = self._load_embedder()
        self.knowledge_base = self._load_sample_knowledge()
        self.index = self._build_index()
        
    @st.cache_resource
    def _load_embedder(_self):
        """Load sentence transformer model."""
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load embedder: {e}")
            return None
    
    def _load_sample_knowledge(self):
        """Load sample knowledge base."""
        return [
            {
                "text": "Our PostgreSQL database supports advanced querying, indexing, and transaction management. It's fully ACID compliant and supports JSON operations.",
                "source": "PostgreSQL Manual",
                "page": 1
            },
            {
                "text": "To reset a forgotten password in PostgreSQL on Ubuntu 22, use: sudo -u postgres psql -c \"ALTER USER username PASSWORD 'newpassword';\"",
                "source": "PostgreSQL Manual", 
                "page": 234
            },
            {
                "text": "Our TechCorp Product Suite includes CRM, ERP, and Analytics modules. All modules integrate seamlessly with REST APIs.",
                "source": "TechCorp Product Manual",
                "page": 15
            },
            {
                "text": "For database connection issues, check: 1) Network connectivity, 2) Firewall settings, 3) PostgreSQL service status, 4) Authentication configuration.",
                "source": "Company FAQ",
                "page": 3
            },
            {
                "text": "Support tickets are automatically assigned based on priority: Critical (1h), High (4h), Medium (24h), Low (72h).",
                "source": "Support Guidelines",
                "page": 7
            }
        ]
    
    def _build_index(self):
        """Build FAISS index from knowledge base."""
        if not self.embedder:
            return None
            
        try:
            texts = [item["text"] for item in self.knowledge_base]
            embeddings = self.embedder.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            return index
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            return None
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents."""
        if not self.embedder or not self.index:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_base):
                    result = self.knowledge_base[idx].copy()
                    result['score'] = float(score)
                    results.append(result)
            
            return results
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with citations."""
        # Search for relevant documents
        search_results = self.search(query)
        
        if not search_results:
            return {
                "answer": "I couldn't find specific information about your question in our knowledge base. Would you like me to create a support ticket for you?",
                "sources": [],
                "confidence": 0.0
            }
        
        # Simple response generation (in a real system, this would use an LLM)
        best_result = search_results[0]
        confidence = best_result['score']
        
        if confidence > 0.5:
            answer = f"Based on our documentation: {best_result['text']}"
            if len(search_results) > 1:
                answer += f"\n\nAdditional information: {search_results[1]['text']}"
        else:
            answer = "I found some potentially relevant information, but I'm not confident it fully answers your question. Would you like me to create a support ticket for you?"
        
        return {
            "answer": answer,
            "sources": search_results,
            "confidence": confidence
        }

def create_support_ticket(title: str, description: str, name: str, email: str) -> str:
    """Simulate support ticket creation."""
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    # In a real system, this would integrate with a ticketing system
    ticket_data = {
        "id": ticket_id,
        "title": title,
        "description": description,
        "name": name,
        "email": email,
        "created_at": datetime.now().isoformat(),
        "status": "Open",
        "priority": "Medium"
    }
    
    return ticket_id

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title=f"{COMPANY_NAME} - Support Chat",
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
    .source-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 0.4rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing RAG chatbot..."):
            st.session_state.chatbot = SimpleRAGChatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Company**: {COMPANY_NAME}
        **Email**: {COMPANY_EMAIL}
        **Phone**: {COMPANY_PHONE}
        **Website**: {COMPANY_WEBSITE}
        """)
        
        st.markdown("---")
        st.header("🎯 Quick Examples")
        sample_queries = [
            "How do I reset a PostgreSQL password?",
            "What products does TechCorp offer?", 
            "How to troubleshoot database connection issues?",
            "What are the support ticket priorities?"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query[:20]}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        st.markdown("---")
        st.header("🎫 Need More Help?")
        if st.button("Create Support Ticket"):
            st.session_state.show_ticket_form = True
            st.rerun()
    
    # Main content
    st.title(f"🤖 {COMPANY_NAME} Support Assistant")
    st.markdown("Ask me anything about our products and services. I can help you find information or create a support ticket.")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                        <strong>{source['source']}</strong> (Page {source['page']}) - Score: {source['score']:.3f}<br>
                        {source['text'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response = st.session_state.chatbot.generate_response(prompt)
            
            st.markdown(response["answer"])
            
            # Show sources
            if response["sources"]:
                with st.expander("📚 Sources"):
                    for source in response["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                        <strong>{source['source']}</strong> (Page {source['page']}) - Score: {source['score']:.3f}<br>
                        {source['text']}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"],
            "sources": response["sources"]
        })
    
    # Support ticket form
    if st.session_state.get('show_ticket_form', False):
        st.markdown("---")
        st.header("🎫 Create Support Ticket")
        
        with st.form("ticket_form"):
            title = st.text_input("Ticket Title", max_chars=100)
            description = st.text_area("Description", height=100, max_chars=500)
            name = st.text_input("Your Name", max_chars=50)
            email = st.text_input("Email Address")
            
            submitted = st.form_submit_button("Create Ticket")
            
            if submitted:
                if title and description and name and email:
                    ticket_id = create_support_ticket(title, description, name, email)
                    st.success(f"✅ Support ticket created successfully!\n\n**Ticket ID**: {ticket_id}")
                    st.session_state.show_ticket_form = False
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")

if __name__ == "__main__":
    main()
