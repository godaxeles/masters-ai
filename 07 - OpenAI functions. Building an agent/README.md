# Database Query Assistant

## Project Description
**Database Query Assistant** is a web application that allows users to query databases using SQL queries, powered by GPT-4. Users can input natural language queries, and the system automatically generates corresponding SQL queries to work with the database. This project integrates the OpenAI API for query execution and SQLite for data storage.

## Key Features
- **Database Queries**: Enables users to input natural language queries that are converted into SQL queries to retrieve information from the database.
- **Result Display**: After executing an SQL query, results are displayed in the interface.
- **Charts**: When necessary, displays charts for data visualization (e.g., price change dynamics).
- **Message History**: All user conversations are saved so the system can consider context in subsequent queries.

## Technology Stack
- **Python** 3.8+
- **Streamlit** — for creating the web interface.
- **OpenAI GPT-4** — for generating SQL queries and processing user messages.
- **SQLite** — for database operations.
- **pandas** and **matplotlib** — for data processing and chart creation.
- **Tenacity** — for API error retries.

## Requirements
Before running the project, make sure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

## Installation and Launch

1. Clone the repository:
```bash
git clone https://github.com/yourusername/database-query-assistant.git
cd database-query-assistant
```

2. Create an `.env` file and add your OpenAI API key:
```plaintext
OPENAI_API_KEY=your_openai_api_key
```

3. Ensure all required libraries are installed:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run main.py
```

5. Open the application in your browser at http://localhost:8501.

## Project Structure
```
.
├── main.py             # Main application script
├── conversation.py     # Logic for conversation handling and message storage
├── requirements.txt    # List of dependencies
├── .env               # Environment variables file (OpenAI API key)
├── README.md          # This file
└── products_data.db   # Sample SQLite database
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
