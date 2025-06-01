# Customer Support RAG Chatbot

A comprehensive Customer Support solution powered by RAG (Retrieval-Augmented Generation) that can answer questions from datasources and create support tickets.

## Features

- 🤖 **Smart Q&A**: Answer questions from company documentation with citations
- 🎫 **Ticket Creation**: Automatically create support tickets when answers aren't found
- 📚 **Document Processing**: Supports PDF, TXT, and Markdown documents
- 🔍 **Vector Search**: Uses FAISS for efficient similarity search
- 💬 **Chat Interface**: Modern Streamlit interface with conversation history
- 📊 **Analytics**: Built-in metrics and performance monitoring

## Technical Stack

- **Framework**: Streamlit + Gradio
- **AI/ML**: sentence-transformers (all-MiniLM-L6-v2), FAISS vector store
- **Document Processing**: pypdf, python-docx
- **Integrations**: GitHub Issues API for ticket management
- **Testing**: pytest with comprehensive unit, integration, and E2E tests

## Live Demo

This demo includes sample documents:
- PostgreSQL Manual (3,044 pages)
- TechCorp Product Manual (117 pages)  
- Company FAQ and guides

## Usage

1. Ask questions about the products or documentation
2. Get answers with source citations
3. Create support tickets for unresolved issues
4. View conversation history and metrics

## Requirements Met

✅ **Functional Requirements:**
- Customer Support Q&A system
- Support ticket creation workflow
- Document citation with source + page
- Conversation history management
- Company information integration

✅ **Technical Requirements:**
- Python-based solution
- Vector storage (FAISS)
- Streamlit/Gradio UI
- 3+ documents (5 documents included)
- 2+ PDFs with 1 having 400+ pages
- Comprehensive test suite (TDD/BDD)

## Configuration

Set these secrets in HuggingFace Spaces:

```
GITHUB_TOKEN=your_github_token
GITHUB_REPO=username/repo
COMPANY_NAME=Your Company Name
COMPANY_EMAIL=support@company.com
COMPANY_PHONE=+1-555-0123
COMPANY_WEBSITE=https://company.com
```

## Local Development

```bash
git clone https://github.com/godaxeles/masters-ai.git
cd Semester-02-Capstone-Project-Customer-ChatBot
pip install -r requirements.txt
python -m src.ingest  # Process documents
streamlit run src/app.py  # Launch app
```
