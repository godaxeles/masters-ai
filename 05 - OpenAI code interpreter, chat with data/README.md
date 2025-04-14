# PDF Chat & Data Visualization

This project implements two main functionalities:
1. A PDF document processing and chat system using OpenAI's API
2. A CSV data visualization tool that creates histograms with matplotlib

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/godaxeles/masters-ai.git
cd pdf-csv-processor
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

4. Create a `.env` file in the project root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Data Sources

### PDF Document

For the PDF document processing, I used the paper "Attention Is All You Need" by Vaswani et al., which is a foundational paper on transformer models in machine learning. The paper has 100+ pages and is publicly available.

- **Source**: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- **Download Command**:
```bash
wget https://arxiv.org/pdf/1706.03762.pdf -O attention_paper.pdf
```

### CSV Dataset

For CSV data visualization, I used the NYC Taxi Trip data, which contains over 1 million rows and multiple columns including pickup/dropoff times, locations, fares, etc.

- **Source**: [NYC Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Download Command** (smaller sample version for testing):
```bash
wget https://github.com/noelrappin/ml_resources/raw/master/yellow_tripdata_2021-01_sample.csv -O taxi_data.csv
```

## Usage

### PDF Chat

To query the PDF document:

```bash
python pdf_chat.py --pdf attention_paper.pdf --query "Explain the attention mechanism in simple terms"
```

This will:
1. Process the PDF document
2. Extract relevant chunks based on the query
3. Send the chunks to the OpenAI API
4. Return the response

### CSV Visualization

To create a histogram from CSV data:

```bash
python csv_visualizer.py --csv taxi_data.csv --column fare_amount --save histogram.png
```

This will:
1. Load the CSV file
2. Create a histogram of the specified column
3. Save the visualization to the specified file

## Example Prompts and Results

### PDF Chat Prompt

```
What are the key components of the transformer architecture?
```

#### Result:
![PDF Chat Result](screenshots/pdf_chat_result.png)

### CSV Visualization Prompt

```
Create a histogram of the fare_amount column from the NYC Taxi data
```

#### Result:
![CSV Visualization Result](screenshots/csv_histogram.png)

## Implementation Details

### PDF Processing

- Used PyPDFLoader to load and extract text from the PDF
- Split documents into manageable chunks with RecursiveCharacterTextSplitter
- Created a vector store using FAISS for semantic search
- Used OpenAI embeddings to create vector representations
- Retrieved the most relevant chunks based on the query
- Sent the context and query to OpenAI's Chat API

### CSV Visualization

- Used pandas to load and process the CSV data
- Analyzed data types and structure
- Created histograms using matplotlib
- Configured visualization parameters (bins, colors, labels)
- Saved visualizations to files

## Extension Ideas

- Add a web interface using Streamlit or Flask
- Implement multiple visualization types (scatter plots, bar charts)
- Support for multiple data sources
- Add data preprocessing options
