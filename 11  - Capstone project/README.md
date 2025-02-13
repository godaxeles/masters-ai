# AI Chatbot for Movie Data with SQL Integration

## Project Overview

This project is a chatbot application that interacts with a SQLite database containing employee information. The bot uses OpenAI's GPT-4 to generate valid SQL queries based on user input. The system is integrated with Streamlit for the user interface, and it handles employee data, sending emails, and logging queries.

Key features:
- Querying employee data through the chatbot.
- Dynamic SQL generation with OpenAI's GPT-4.
- Filterable employee data using Streamlit's interface.
- Salary distribution visualization via pie charts.

## Project Requirements

- Python 3.12 or below
- Dependencies:
  - OpenAI
  - SQLite3 (included with Python)
  - Pandas
  - Streamlit
  - Faker
  - Matplotlib

### Installation

1. Clone or download this repository.
2. Install the required dependencies using `pip`:
   ```bash
   pip install openai requests termcolor streamlit python-dotenv pandas matplotlib faker
