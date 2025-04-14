import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
import logging
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_openai_client():
    """Initialize OpenAI client with API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    
def process_pdf(pdf_path):
    """Process PDF document and create a FAISS index for searching."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split PDF into {len(chunks)} chunks")
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Created FAISS index from PDF chunks")
    
    return vectorstore

def query_pdf(vectorstore, query, client):
    """Query PDF using vector search and OpenAI completion."""
    logger.info(f"Querying PDF with: {query}")
    
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate response using OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context."},
        {"role": "user", "content": f"Context from document:\n\n{context}\n\nQuestion: {query}\n\nAnswer the question based only on the provided context:"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def analyze_csv(csv_path, column_name=None, save_path=None):
    """Analyze CSV data and create histogram for specified column."""
    logger.info(f"Analyzing CSV: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names and data types:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    
    # Select column for histogram if not specified
    if column_name is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the CSV")
        column_name = numeric_cols[0]
        logger.info(f"No column specified, using: {column_name}")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df[column_name].dropna(), bins=25, edgecolor='black', color='skyblue')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Save or show histogram
    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path)
        logger.info(f"Histogram saved to: {save_path}")
    else:
        plt.show()
    
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF and CSV data')
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--column', type=str, help='Column name for histogram')
    parser.add_argument('--query', type=str, help='Query to search in PDF')
    parser.add_argument('--save', type=str, help='Path to save histogram')
    args = parser.parse_args()
    
    # Process based on provided arguments
    client = setup_openai_client()
    
    if args.pdf and args.query:
        vectorstore = process_pdf(args.pdf)
        response = query_pdf(vectorstore, args.query, client)
        print(f"\nQuery: {args.query}")
        print(f"\nResponse:\n{response}")
    
    if args.csv:
        analyze_csv(args.csv, args.column, args.save)

if __name__ == "__main__":
    main()
