# Business Intelligence Agent

This project implements an AI agent that retrieves information from multiple datasources (database and documents) and presents it through a user-friendly Streamlit interface.

## Features

- **Database Integration**: Queries a SQLite database with business information
- **Document Retrieval**: Uses FAISS for semantic search of relevant document chunks
- **LLM Integration**: Only sends relevant data to the LLM, not the entire datasource
- **Streamlit UI**: User-friendly interface with chat functionality
- **API Integration**: Ability to call external APIs for various business actions
- **Business Dashboard**: Displays key business metrics in the sidebar
- **Comprehensive Logging**: Tracks all operations for monitoring and debugging
- **Security Features**: SQL injection prevention and input validation

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/godaxeles/masters-ai.git
cd business-intelligence-agent
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
API_BASE_URL=https://api.example.com
API_KEY=your_api_key_here
```

5. Set up the database and sample documents:
```bash
python setup_database.py
python create_documents.py
```

## Data Sources

### Database

The project uses a SQLite database with the following tables:
- `business_info`: General company information
- `products`: Product catalog
- `orders`: Customer orders
- `order_items`: Items within each order

### Documents

Sample business documents in the `/data/documents` directory:
- Company policies
- Product manuals
- Customer service guidelines
- Sales strategies

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

This will start the web interface on http://localhost:8501

## Example Queries and Results

### Database Query

```
What are our top-selling products?
```

#### Result:
![Database Query Result](screenshots/db_query_result.png)

### Document Query

```
What is our remote work policy?
```

#### Result:
![Document Query Result](screenshots/doc_query_result.png)

### API Call

```
Action: Create Ticket
Parameters: {"priority": "high", "subject": "Server outage", "description": "Customer reporting server connectivity issues"}
```

#### Result:
![API Call Result](screenshots/api_call_result.png)

## Implementation Details

### Database Integration

- Secure parameterized SQL queries to prevent injection attacks
- Efficient data retrieval with optimized queries
- Data transformation for LLM consumption

### Document Retrieval

- Document loading and chunking for efficient processing
- Vector embeddings using OpenAI Embeddings
- FAISS vector store for fast similarity search
- Relevance-based chunk selection

### Query Processing

- Intelligent source determination (database, documents, or both)
- Context compilation from multiple sources
- Structured prompt engineering for optimal LLM responses

### External API Integration

- Secure API calls with proper error handling
- JSON validation for request payloads
- Comprehensive logging of API interactions

### Security Measures

- SQL injection prevention through input sanitization
- API request validation and error handling
- Environment variable management for secrets

## Project Structure

```
business-intelligence-agent/
├── app.py                 # Main Streamlit application
├── setup_database.py      # Database setup script
├── create_documents.py    # Document creation script
├── data/                  # Data directory
│   ├── business_data.db   # SQLite database
│   └── documents/         # Document files
├── .env                   # Environment variables
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Logging

Logs are saved to `app.log` and include information about:
- Database queries
- Document retrievals
- API calls
- LLM interactions
- User queries

## Future Enhancements

- Integration with more data sources
- Advanced NLP for better query understanding
- More sophisticated visualization capabilities
- User authentication and role-based access control
