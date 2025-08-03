"""
Streamlit web application for the RAG system with ticket creation integration.
"""

import streamlit as st
import os
import logging
from dotenv import load_dotenv
import requests # For Trello API or potentially Jira
# from jira import JIRA # Install if Jira is chosen

# --- Import RAG Core Logic --- 
# Assuming RAG_chatbot.py is in the same directory or accessible in PYTHONPATH
# We will copy/adapt necessary functions directly into this file for simplicity in this context,
# but using it as a module would be better practice in a real project.

# --- Logging Configuration --- 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Load Environment Variables --- 
load_dotenv() # Load variables from .env file

# --- RAG Configuration (Copied/Adapted from RAG_chatbot.py) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_BASE_FOLDER = "./data"
VECTOR_STORE_DIR = "./vector_store"
DOCS_IN_RETRIEVER = 15
RELEVANCE_THRESHOLD_DOCS = 0.7
RELEVANCE_THRESHOLD_PROMPT = 0.6

# --- Ticket System Configuration --- 
# Set TICKET_SYSTEM to 'Trello' (or 'Jira' if configured)
# Ensure this is set in your .env file or environment
TICKET_SYSTEM = os.getenv("TICKET_SYSTEM", "Trello") 

# Jira Specific (Example Env Vars - Required if TICKET_SYSTEM='Jira')
JIRA_URL = os.getenv("JIRA_URL")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
JIRA_ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Trello Specific (Example Env Vars - Required if TICKET_SYSTEM='Trello')
# Ensure these are set in your .env file or environment
# TRELLO_API_KEY=your_trello_api_key
# TRELLO_API_TOKEN=your_trello_api_token
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID") # Defaulting for clarity, but env var is preferred
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID") # Defaulting for clarity, but env var is preferred
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_API_TOKEN = os.getenv("TRELLO_API_TOKEN")

# --- RAG Core Functions (Copied/Adapted from RAG_chatbot.py) ---

# Need to import these if not copying functions directly
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage # For chat history

# Initialize LLM and Embeddings globally or within a cached function for Streamlit
@st.cache_resource # Cache resource across reruns
def get_llm_and_embeddings():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in your environment variables or .env file.")
        st.stop()
    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return llm, embeddings
    except Exception as e:
        st.error(f"Failed to initialize OpenAI models: {e}")
        st.stop()

llm, embeddings = get_llm_and_embeddings()

@st.cache_resource # Cache the vector store loading/creation
def load_or_create_vector_store(_embeddings): # Pass embeddings explicitly
    """Loads existing FAISS store or creates a new one if needed."""
    index_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_DIR, 
                _embeddings, 
                allow_dangerous_deserialization=True 
            )
            logging.info(f"Vector store loaded from: {VECTOR_STORE_DIR}")
            return vector_store
        except Exception as e:
            logging.error(f"Error loading vector store from {VECTOR_STORE_DIR}: {e}")
            # Fall through to creation if loading fails
    
    logging.info(f"No existing vector store found or load failed in {VECTOR_STORE_DIR}. Indexing documents from {KNOWLEDGE_BASE_FOLDER}...")
    documents = []
    if not os.path.isdir(KNOWLEDGE_BASE_FOLDER):
        st.error(f"Knowledge base folder not found: {KNOWLEDGE_BASE_FOLDER}")
        st.stop()

    pdf_files_found = False
    # Use st.progress for feedback during indexing
    progress_bar = st.progress(0, text="Scanning PDF files...")
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_FOLDER) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)

    for i, filename in enumerate(pdf_files):
        progress_bar.progress((i + 1) / (total_files + 1), text=f"Loading {filename}...")
        pdf_files_found = True
        pdf_path = os.path.join(KNOWLEDGE_BASE_FOLDER, filename)
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
            logging.info(f"Loaded {len(pdf_docs)} pages from {filename}")
        except Exception as e:
            logging.error(f"Error reading or loading {filename}: {e}")
            st.warning(f"Could not load {filename}: {e}")

    if not pdf_files_found:
        st.error(f"No PDF files found in {KNOWLEDGE_BASE_FOLDER}. Cannot create vector store.")
        st.stop()
    if not documents:
        st.error("No documents successfully loaded from PDF files. Cannot create vector store.")
        st.stop()

    progress_bar.progress(0.8, text="Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    except Exception as e:
        st.error(f"Error splitting documents: {e}")
        st.stop()

    progress_bar.progress(0.9, text="Creating vector index...")
    try:
        vector_store = FAISS.from_documents(split_docs, _embeddings)
        logging.info("Documents successfully indexed in FAISS.")
        # Save the newly created store
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_DIR)
        logging.info(f"Vector store saved to: {VECTOR_STORE_DIR}")
        progress_bar.progress(1.0, text="Indexing complete!")
        progress_bar.empty() # Remove progress bar
        st.success("Document indexing complete and vector store created.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        st.stop()

# --- RAG Generation Logic (Adapted for Streamlit) --- 

# Note: Pre- and post-processing with LLM adds latency. Consider simplifying.
def preprocess_user_prompt(user_prompt: str, chat_history: list, llm) -> str:
    # Simplified version for Streamlit - can add LLM call back if needed
    logging.info(f"Preprocessing prompt: 	{user_prompt}")
    return user_prompt 

def postprocess_llm_response(llm_response: str, user_prompt: str, references: dict = None, is_relevant: bool = False) -> tuple:
    # Simplified version for Streamlit - can add LLM call back if needed
    logging.info("Post-processing LLM response.")
    final_answer = llm_response
    if is_relevant and references:
        ref_texts = []
        for source, pages in references.items():
            page_str = ", ".join(map(str, sorted(list(pages))))
            ref_texts.append(f"- {source} (Page(s): {page_str})")
        if ref_texts:
            final_answer += "\n\n**References:**\n" + "\n".join(ref_texts)
            
    return final_answer, references

def retrieve_documents_with_similarity(vector_store, user_prompt: str, k: int = DOCS_IN_RETRIEVER):
    if not vector_store:
        logging.error("Vector store not available for retrieval.")
        return []
    try:
        docs_with_scores = vector_store.similarity_search_with_relevance_scores(user_prompt, k=k)
        logging.info(f"Retrieved {len(docs_with_scores)} documents with relevance scores.")
        return docs_with_scores
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        st.warning(f"Error during document retrieval: {e}")
        return []

def is_prompt_relevant_to_documents(relevance_scores, relevance_threshold=RELEVANCE_THRESHOLD_PROMPT):
    if not relevance_scores:
        return False
    try:
        max_similarity = max((score for _, score in relevance_scores), default=0.0)
        is_relevant = max_similarity >= relevance_threshold
        logging.info(f"Max similarity score: {max_similarity:.4f}, Threshold: {relevance_threshold}, Prompt relevant?: {is_relevant}")
        return is_relevant
    except Exception as e:
        logging.error(f"Exception in is_prompt_relevant_to_documents: {e}")
        return False

# Main RAG function adapted for Streamlit
# Note: Still uses sync LLM calls. For high concurrency, use async versions.
def run_rag_query(vector_store, user_prompt: str, chat_history_langchain: list):
    """Runs the RAG pipeline and returns the answer and sources."""
    
    prepared_prompt = preprocess_user_prompt(user_prompt, chat_history_langchain, llm)
    
    retrieved_docs_with_scores = retrieve_documents_with_similarity(
        vector_store=vector_store,
        user_prompt=prepared_prompt,
        k=DOCS_IN_RETRIEVER
    )
    
    relevant_docs_with_scores = [
        (doc, score) for doc, score in retrieved_docs_with_scores
        if score >= RELEVANCE_THRESHOLD_DOCS
    ]
    relevant_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_relevant = 5 
    relevant_docs = [doc for doc, _ in relevant_docs_with_scores[:top_k_relevant]]
    logging.info(f"Filtered down to {len(relevant_docs)} relevant documents.")

    is_relevant = is_prompt_relevant_to_documents(retrieved_docs_with_scores, RELEVANCE_THRESHOLD_PROMPT)

    if not relevant_docs or not is_relevant:
        logging.info("No relevant documents found or prompt deemed irrelevant.")
        # Return a specific indicator for fallback
        return "FALLBACK_NO_INFO", None, None 

    context_str = ""
    references = {}
    for i, doc in enumerate(relevant_docs):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content or "No Content"
        context_str += f"--- Document {i+1} (Source: {os.path.basename(source)}, Page: {page}) ---\n{content}\n\n"
        try: 
            page_num = int(page) + 1 # PyPDFLoader pages are 0-indexed
        except (ValueError, TypeError):
            page_num = page
        references.setdefault(os.path.basename(source), set()).add(page_num)
        
    formatted_references = { 
        source: sorted(list(pages)) 
        for source, pages in references.items() 
    }

    system_prompt = (
        "You are an expert assistant. Provide a clear and concise answer based ONLY on the provided context documents below. "
        "Do not use any prior knowledge. If the answer is not found in the context, state that clearly.\n\n"
        "--- Context Documents ---\n"
        f"{context_str}"
        "--- End Context ---\n"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt_template | llm
    try:
        # Use prepared prompt for the actual query to LLM
        llm_result = chain.invoke({"input": prepared_prompt, "chat_history": chat_history_langchain})
        answer_text = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
        logging.info("Received response from LLM based on context.")
    except Exception as e:
        logging.error(f"Error invoking LLM chain: {e}")
        st.error(f"Error generating response: {e}")
        # Return specific indicator for error
        return "FALLBACK_ERROR", None, None

    final_answer, _ = postprocess_llm_response(
        llm_response=answer_text,
        user_prompt=user_prompt, # Use original prompt for final context
        references=formatted_references,
        is_relevant=True
    )

    source_files = list(formatted_references.keys()) if formatted_references else None
    return final_answer, source_files, formatted_references

# --- Ticket Creation Functions --- 

def create_jira_ticket(summary: str, description: str, user_name: str, user_email: str):
    """Creates a ticket in Jira (Requires configuration and `jira` library)."""
    st.info(f"Jira Integration: Attempting to create ticket...\nUser: {user_name} ({user_email})\nSummary: {summary}")
    # Check for required Jira configuration from environment variables
    if not all([JIRA_URL, JIRA_PROJECT_KEY, JIRA_USER_EMAIL, JIRA_API_TOKEN]):
        st.error("Jira configuration is incomplete. Cannot create ticket. Please set JIRA_URL, JIRA_PROJECT_KEY, JIRA_USER_EMAIL, and JIRA_API_TOKEN environment variables.")
        return False
    
    try:
        # Example using python-jira library (needs installation: pip install jira)
        from jira import JIRA
        auth_jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USER_EMAIL, JIRA_API_TOKEN))
        issue_dict = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": f"Reported by: {user_name} ({user_email})\n\n{description}",
            "issuetype": {"name": JIRA_ISSUE_TYPE},
            # Add other fields as needed, e.g., labels, components
        }
        new_issue = auth_jira.create_issue(fields=issue_dict)
        st.success(f"Successfully created Jira issue: {new_issue.key}")
        logging.info(f"Created Jira issue {new_issue.key}")
        return True
        # st.warning("Jira ticket creation logic is commented out. Requires `jira` library and full configuration.")
        # return False # Placeholder
    except ImportError:
        st.error("Jira library not installed. Please run `pip install jira`.")
        logging.error("Jira library not found.")
        return False
    except Exception as e:
        st.error(f"Failed to create Jira ticket: {e}")
        logging.error(f"Jira ticket creation failed: {e}")
        return False

def create_trello_ticket(summary: str, description: str, user_name: str, user_email: str):
    """Creates a card in Trello using environment variables for configuration."""
    st.info(f"Trello Integration: Attempting to create card...\nUser: {user_name} ({user_email})\nSummary: {summary}")
    # Check for required Trello configuration from environment variables
    # TRELLO_BOARD_ID is not strictly needed for card creation, only LIST_ID, KEY, TOKEN
    if not all([TRELLO_LIST_ID, TRELLO_API_KEY, TRELLO_API_TOKEN]):
        st.error("Trello configuration is incomplete. Cannot create card. Please set TRELLO_LIST_ID, TRELLO_API_KEY, and TRELLO_API_TOKEN environment variables.")
        return False

    try:
        url = f"https://api.trello.com/1/cards"
        query = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "idList": TRELLO_LIST_ID, # Use the List ID from env var
            "name": summary,
            "desc": f"Reported by: {user_name} ({user_email})\n\n{description}"
            # Add other parameters like idMembers, labels, etc. if needed
        }
        response = requests.post(url, params=query)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        card_info = response.json()
        card_id = card_info["id"]
        card_url = card_info["shortUrl"]
        st.success(f"Successfully created Trello card: [{card_id}]({card_url})", icon="✅")
        logging.info(f"Created Trello card {card_id}")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create Trello card: API request failed - {e}")
        logging.error(f"Trello card creation failed: {e}")
        return False
    except Exception as e:
        st.error(f"Failed to create Trello card: An unexpected error occurred - {e}")
        logging.error(f"Trello card creation failed: {e}")
        return False

# --- Streamlit UI --- 

st.set_page_config(page_title="Document RAG Chat", layout="wide")
st.title("📄 Document RAG Chatbot")

# Sidebar for configuration and information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application allows you to chat with your PDF documents using RAG (Retrieval-Augmented Generation) technology.
    
    If the answer to your question isn't found in the documents, you can create a support ticket.
    """)
    
    st.header("Configuration")
    # Display the configured ticket system
    if TICKET_SYSTEM.lower() == "jira":
        st.markdown(f"**Ticket System:** Jira")
        if not all([JIRA_URL, JIRA_PROJECT_KEY, JIRA_USER_EMAIL, JIRA_API_TOKEN]):
             st.warning("Jira environment variables (URL, PROJECT_KEY, USER_EMAIL, API_TOKEN) seem incomplete.")
    elif TICKET_SYSTEM.lower() == "trello":
        st.markdown(f"**Ticket System:** Trello")
        if not all([TRELLO_LIST_ID, TRELLO_API_KEY, TRELLO_API_TOKEN]):
             st.warning("Trello environment variables (LIST_ID, API_KEY, API_TOKEN) seem incomplete.")
    else:
        st.markdown(f"**Ticket System:** Not Configured (Set TICKET_SYSTEM to 'Jira' or 'Trello')")

    # Upload PDF files
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files to './data' folder", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        files_processed = True
        with st.spinner("Processing uploaded files..."):
            os.makedirs(KNOWLEDGE_BASE_FOLDER, exist_ok=True)
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join(KNOWLEDGE_BASE_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
                    files_processed = False
            
            if files_processed:
                # Reset vector store to force reindexing
                if "vector_store_loaded" in st.session_state:
                    st.session_state.vector_store_loaded = False
                    # Clear cached resource if necessary (may require Streamlit >= 1.18)
                    # st.cache_resource.clear()
                    # For older versions, manually delete the cache or restart
                st.info("Files uploaded. Re-indexing will occur on next query or page refresh.")
                # Consider adding a button to trigger re-indexing explicitly
                # st.rerun() # Rerun might be too disruptive if user is typing

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ticket_info_requested" not in st.session_state:
    st.session_state.ticket_info_requested = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Load vector store once or if triggered by upload
if not st.session_state.vector_store_loaded:
    with st.spinner("Loading/Indexing documents... This may take a while the first time."):
        st.session_state.vector_store = load_or_create_vector_store(embeddings)
        if st.session_state.vector_store:
            st.session_state.vector_store_loaded = True
        else:
            st.error("Failed to load or create the vector store. Cannot proceed.")
            st.stop()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle ticket creation state
if st.session_state.ticket_info_requested:
    st.info("I couldn't find the answer in the documents. Please provide details to create a support ticket.")
    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Your Name *", key="t_name")
        with col2:
            user_email = st.text_input("Your Email *", key="t_email")
        
        summary = st.text_input("Ticket Summary (Title) *", value=f"Query: {st.session_state.last_query}", key="t_summary")
        description = st.text_area("Ticket Description *", value=f"User query: {st.session_state.last_query}\n\nDetails: [Please add more details about the issue or question]", key="t_desc")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("Create Ticket")
        
        if submitted:
            if not user_name or not user_email or not summary or not description:
                st.warning("Please fill in all required fields (*) to create a ticket.")
            else:
                success = False
                with st.spinner(f"Creating {TICKET_SYSTEM} ticket..."):
                    if TICKET_SYSTEM.lower() == "jira":
                        success = create_jira_ticket(summary, description, user_name, user_email)
                    elif TICKET_SYSTEM.lower() == "trello":
                        success = create_trello_ticket(summary, description, user_name, user_email)
                    else:
                        st.error("Ticket system not configured or supported. Please check TICKET_SYSTEM environment variable.")
                
                if success:
                    # Add confirmation to chat
                    ticket_confirm_msg = f"Support ticket created successfully in {TICKET_SYSTEM}. The support team will review it." 
                    st.session_state.messages.append({"role": "assistant", "content": ticket_confirm_msg})
                    st.session_state.ticket_info_requested = False # Reset state
                    st.rerun() # Rerun to display the confirmation and hide form
                # Error messages handled within creation functions

# Main chat input
if not st.session_state.ticket_info_requested:
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare chat history for LangChain (list of HumanMessage/AIMessage)
        chat_history_langchain = []
        for msg in st.session_state.messages[:-1]: # Exclude the current prompt
            if msg["role"] == "user":
                chat_history_langchain.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_langchain.append(AIMessage(content=msg["content"]))

        # Generate response using RAG
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            response, sources, references = run_rag_query(
                st.session_state.vector_store, 
                prompt, 
                chat_history_langchain
            )

            # Handle fallback cases
            if response == "FALLBACK_NO_INFO":
                fallback_msg = "I couldn't find relevant information in the documents to answer your question. Would you like to create a support ticket?"
                message_placeholder.markdown(fallback_msg)
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                st.session_state.last_query = prompt # Store query for ticket form
                st.session_state.ticket_info_requested = True
                # Delay rerun slightly to allow user to see the message
                # asyncio.sleep(1) # Requires running streamlit with asyncio patch or handling differently
                st.rerun() # Rerun to show the ticket form
            elif response == "FALLBACK_ERROR":
                assistant_response = "Sorry, I encountered an error while trying to answer your question." 
                message_placeholder.markdown(assistant_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                message_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

# Add a footer
st.markdown("---")
st.markdown("*Powered by LangChain, FAISS, OpenAI, and Streamlit*")

