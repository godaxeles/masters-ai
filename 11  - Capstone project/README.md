# Real Estate Chatbot & Data Viewer

This project is a **Streamlit-based** web application that allows users to interact with a **real estate database** using **SQL queries** and **OpenAI's API**. The app displays queried data in a dataframe and a map, while also providing a chatbot interface for natural language interactions.

## Features

- **SQLite Database Integration**: Queries and retrieves real estate listings.
- **Interactive Map**: Displays property locations using latitude and longitude.
- **Chatbot Interface**: Allows users to query real estate data using natural language.
- **Currency Conversion**: Converts property prices into different currencies using `freecurrencyapi`.
- **Automated SQL Query Handling**: Converts natural language queries into SQL commands.

## Technologies Used

- **Python**
- **Streamlit**
- **SQLite**
- **OpenAI API**
- **freecurrencyapi**
- **Pandas**
- **Dotenv**
- **Tenacity** (for retry logic)

## Installation

### Prerequisites

Make sure you have **Python 3.8+** installed on your system.

### Setup

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and set your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   FREECURRENCYAPI_KEY=your_freecurrencyapi_key
   MY_DATABASE=your_database_path
   ```

## Usage

### Running the App

Run the Streamlit app using:

```bash
streamlit run app.py
```

### Querying the Database

- Default query: `SELECT * FROM buildings_with_coordinates ORDER BY project_no ASC;`
- Users can enter **natural language queries**, which will be converted to SQL.
- The chatbot assists in retrieving and converting data.

### Currency Conversion

- The app supports currency conversion using `freecurrencyapi`.
- Supported currencies include **USD, EUR, JPY, GBP, AUD, CAD, etc.**

## Database Schema

The `buildings_with_coordinates` table includes fields such as:

- `project_no`: Unique identifier for each project.
- `city`: City name where the building is located.
- `building_name`: Name of the building or complex.
- `price_per_sqm`: Price per square meter.
- `latitude`, `longitude`: Coordinates for mapping.

For the full schema, refer to the `database_schema` section in the code.

## API Integration

- **OpenAI API**: Generates SQL queries from natural language inputs.
- **freecurrencyapi**: Retrieves live exchange rates.

![Real Estate Chatbot](./Real%20estate%20chatbot.png)  

