import streamlit as st
import sqlite3
import json
import logging
import os
import requests
import re
from datetime import datetime
from pathlib import Path
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import sys
import pandas as pd
# from restrictedpython import compile_restricted, safe_builtins
# from restrictedpython.restricted import RestrictedPython


from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("business_agent")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
api_base_url = os.getenv("API_BASE_URL", "https://api.example.com")
api_key = os.getenv("API_KEY", "default_key")

# Initialize global variables
vectorstore = None
db_path = "data/business_data.db"

# Function to sanitize SQL inputs
def sanitize_sql_input(input_string):
    """Sanitize user input for SQL queries to prevent injection attacks."""
    if not isinstance(input_string, str):
        return input_string
    # Remove SQL comment syntax
    sanitized = re.sub(r'--.*$', '', input_string)
    # Remove other common SQL injection patterns
    sanitized = re.sub(r';.*$', '', sanitized)
    sanitized = re.sub(r'\/\*.*?\*\/', '', sanitized)
    return sanitized

# Function to execute secure database queries
def query_database(query, params=None):
    """Execute SQL query with parameters to prevent SQL injection."""
    try:
        logger.info(f"Executing database query: {query}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if params:
            results = cursor.execute(query, params).fetchall()
        else:
            results = cursor.execute(query).fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        conn.close()
        logger.info(f"Database query returned {len(results)} results")
        
        # Convert results to list of dictionaries with column names
        if column_names:
            results_dicts = []
            for row in results:
                results_dicts.append(dict(zip(column_names, row)))
            return results_dicts
        return results
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        return {"error": f"Database error: {str(e)}"}

# Function to get business information
def get_business_info():
    """Retrieve general business information from the database."""
    logger.info("Fetching business information")
    query = """
    SELECT 
        name, 
        industry, 
        revenue, 
        employees,
        founded,
        headquarters,
        customer_satisfaction
    FROM business_info 
    LIMIT 1
    """
    result = query_database(query)
    return result[0] if result else {}

# Function to get pending orders
def get_pending_orders():
    """Retrieve count of pending orders from the database."""
    logger.info("Fetching pending orders count")
    query = "SELECT COUNT(*) AS pending_count FROM orders WHERE status = 'pending'"
    result = query_database(query)
    return result[0]['pending_count'] if result else 0

# Function to initialize document retrieval system
def initialize_document_store(docs_dir="data/documents"):
    """Initialize the document vector store for semantic search."""
    global vectorstore
    
    if vectorstore:
        return vectorstore
    
    logger.info(f"Initializing document store from: {docs_dir}")
    
    try:
        # Create directory if it doesn't exist
        Path(docs_dir).mkdir(parents=True, exist_ok=True)
        
        documents = []
        for file_path in Path(docs_dir).glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
        
        if not documents:
            logger.warning("No documents found in directory")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Created FAISS index for document retrieval")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing document store: {str(e)}")
        return None

# Function to retrieve relevant documents
def retrieve_documents(query, k=3):
    """Retrieve relevant document chunks based on semantic similarity."""
    global vectorstore
    
    logger.info(f"Retrieving documents for query: {query}")
    
    if not vectorstore:
        vectorstore = initialize_document_store()
        
    if not vectorstore:
        logger.error("Document store not initialized")
        return []
    
    try:
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        # Extract relevant information
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
            })
        
        logger.info(f"Retrieved {len(results)} relevant document chunks")
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

# Function to call external API
def call_external_api(action, parameters):
    """Call an external API with the specified action and parameters."""
    logger.info(f"Calling external API: {action} with parameters: {parameters}")
    
    # Validate input
    if not isinstance(parameters, dict):
        try:
            # Attempt to parse JSON string
            parameters = json.loads(parameters)
        except:
            parameters = {"data": parameters}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        url = f"{api_base_url}/{action}"
        
        # Add timeout and error handling
        response = requests.post(
            url, 
            json=parameters, 
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"API call successful: {action}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {str(e)}")
        return {"error": f"API request failed: {str(e)}"}

# Function to determine which data source to query
def determine_data_source(query):
    """Determine whether to query database, documents, or both based on the query."""
    logger.info(f"Determining data source for query: {query}")
    
    # Keywords suggesting database query
    db_keywords = ["sales", "revenue", "customer", "order", "employee", "product", "price", "stock", "inventory"]
    
    # Keywords suggesting document query
    doc_keywords = ["policy", "procedure", "guide", "instruction", "report", "document", "manual", "how to"]
    
    # Check for database keywords
    db_relevance = any(keyword in query.lower() for keyword in db_keywords)
    
    # Check for document keywords
    doc_relevance = any(keyword in query.lower() for keyword in doc_keywords)
    
    # Determine which sources to query
    if db_relevance and doc_relevance:
        logger.info("Query relevant to both database and documents")
        return "both"
    elif db_relevance:
        logger.info("Query relevant to database")
        return "database"
    else:
        logger.info("Query relevant to documents")
        return "documents"

# Function to query LLM
def query_llm(query, context):
    """Query the LLM with the user query and retrieved context."""
    logger.info("Querying LLM with context and user query")
    
    try:
        # Format context for the prompt
        formatted_context = ""
        
        if "database_results" in context:
            formatted_context += "\nDatabase information:\n"
            formatted_context += json.dumps(context["database_results"], indent=2)
        
        if "document_results" in context:
            formatted_context += "\nDocument information:\n"
            for i, doc in enumerate(context["document_results"]):
                formatted_context += f"\nDocument chunk {i+1}:\n{doc['content']}\n"
        
        # Create the message for the LLM
        messages = [
            {"role": "system", "content": "You are a business intelligence assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain relevant information, say so."},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        logger.info("Received response from LLM")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return f"Sorry, I encountered an error processing your request: {str(e)}"

# Function to handle user queries
def process_user_query(query):
    """Process a user query by determining source, retrieving data, and generating response."""
    logger.info(f"Processing user query: {query}")
    
    # Determine which data source to query
    source = determine_data_source(query)
    
    # Initialize context dictionary
    context = {}
    
    # Query the appropriate data sources
    if source in ["database", "both"]:
        # For simplicity, we'll use a general query here
        # In a real system, you would use NLP to determine specific tables/fields
        db_query = """
        SELECT 
            p.name, 
            p.description, 
            p.price, 
            p.stock,
            SUM(o.quantity) as total_ordered
        FROM 
            products p
        LEFT JOIN 
            order_items o ON p.id = o.product_id
        GROUP BY 
            p.id
        LIMIT 10
        """
        database_results = query_database(db_query)
        context["database_results"] = database_results
    
    if source in ["documents", "both"]:
        document_results = retrieve_documents(query)
        context["document_results"] = document_results
    
    # Query the LLM with the retrieved context
    response = query_llm(query, context)
    
    logger.info("Completed processing user query")
    return response

# Function to execute user-provided code
def execute_user_code(code, db_path):
    st.subheader("Code Execution Output")
    
    # Create a dummy context for execution
    # In a real scenario, you'd want to provide safe access to data
    # For now, we'll provide a simple way to query the database
    
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    try:
        # Define a safe environment for execution (without restrictedpython)
        _globals_ = {
            'plt': plt,
            'st': st,
            'pd': pd, # Assuming pandas is imported and available
            'query_database': query_database # Allow access to the query_database function
        }

        # Execute the code directly (without restrictedpython)
        exec(code, _globals_)
        
        # Capture and display matplotlib figures
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free memory
        
        output = redirected_output.getvalue()
        if output:
            st.code(output)
        
    except Exception as e:
        st.error(f"Error during code execution: {e}")
    finally:
        sys.stdout = old_stdout # Restore stdout

# Streamlit UI
def main():
    st.set_page_config(page_title="Business Intelligence Agent", layout="wide")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Page title
    st.title("Business Intelligence Agent")
    
    # Sidebar with business information
    with st.sidebar:
        st.header("Business Dashboard")
        
        try:
            # Get business info
            business_info = get_business_info()
            pending_orders = get_pending_orders()
            
            # Display business information
            st.subheader("Business Information")
            st.write(f"**Name:** {business_info.get('name', 'N/A')}")
            st.write(f"**Industry:** {business_info.get('industry', 'N/A')}")
            st.write(f"**Revenue:** ${business_info.get('revenue', 0):,.2f}")
            st.write(f"**Employees:** {business_info.get('employees', 0)}")
            
            # Display metrics
            st.subheader("Business Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Pending Orders", pending_orders)
            col2.metric("Customer Satisfaction", f"{business_info.get('customer_satisfaction', 0):.1f}/5.0")
            
            # Add last updated timestamp
            st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            st.error(f"Error loading business information: {str(e)}")
    
    # Main area for chat interface
    st.header("Chat with Business Agent")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Agent:** {message['content']}")
        st.write("---")
    
    # Input for user query
    user_query = st.text_input("Ask a question about the business:")
    
    # API action section
    st.header("External API Actions")
    col1, col2 = st.columns(2)
    with col1:
        action = st.selectbox("Select Action:", ["Create Ticket", "Update CRM", "Generate Report", "Send Notification"])
    with col2:
        params = st.text_area("Parameters (JSON format):", "{\"priority\": \"high\"}")
    
    # Process user query
    if st.button("Submit Question"):
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get response
            response = process_user_query(user_query)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the displayed chat
            st.experimental_rerun()
    
    # Process API action
    if st.button("Execute API Action"):
        try:
            # Validate JSON
            params_dict = json.loads(params)
            
            # Call API
            result = call_external_api(action, params_dict)
            
            # Display result
            st.success("API call executed successfully!")
            st.json(result)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in parameters")
        except Exception as e:
            st.error(f"Error executing API action: {str(e)}")

    st.markdown("--- ")
    st.header("📊 Code Interpreter (Experimental)")
    st.warning("⚠️ This feature executes Python code directly. Use with caution and only with trusted code.")
    
    code_input = st.text_area("Enter Python code to execute (e.g., for plotting):", height=200, value="""
import pandas as pd
import matplotlib.pyplot as plt

# Example: Get product data and plot a histogram of prices
products = query_database("SELECT name, price FROM products")
if products:
    df = pd.DataFrame(products)
    plt.figure(figsize=(8, 6))
    plt.hist(df['price'], bins=5, edgecolor='black')
    plt.title('Product Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Number of Products')
    plt.grid(axis='y', alpha=0.75)
    st.write("Histogram generated successfully!")
else:
    st.write("No product data found to plot.")
""")
    
    if st.button("Execute Code"):
        if code_input:
            execute_user_code(code_input, db_path)
        else:
            st.warning("Please enter some code to execute.")

# Entry point
if __name__ == "__main__":
    main()
