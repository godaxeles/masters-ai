# Document RAG Chatbot with Streamlit and Ticket Integration

This application provides an interactive web interface built with Streamlit to chat with your PDF documents using a Retrieval-Augmented Generation (RAG) system. It leverages LangChain, FAISS, and OpenAI to provide answers based on the content of the documents. If the system cannot find a relevant answer in the documents, it offers the user the option to create a support ticket in either Trello or Jira.

## Features

*   **Interactive Chat UI:** A user-friendly chat interface powered by Streamlit.
*   **PDF Document Processing:** Automatically loads and indexes PDF files from a specified directory (`./data`).
*   **Vector Store:** Uses FAISS for efficient similarity searches and stores the index locally (`./vector_store`).
*   **RAG Pipeline:** Retrieves relevant document chunks based on user queries and generates answers using an OpenAI LLM (GPT-4o by default).
*   **Document Upload:** Allows users to upload new PDF documents directly through the sidebar.
*   **Ticket Creation Fallback:** If the RAG system determines that the documents do not contain a relevant answer, it prompts the user to create a support ticket.
*   **Configurable Ticket System:** Supports creating tickets in either Trello or Jira, configured via environment variables.

## Setup

1.  **Clone or Download:** Get the application code, including `streamlit_app_fixed.py` and `requirements.txt`.

2.  **Create Virtual Environment:** It is highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Install the required Python libraries using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you plan to use the Jira integration, uncomment `jira` in `requirements.txt` or install it separately: `pip install jira`.*

4.  **Create `.env` File:** Create a file named `.env` in the root directory of the project. This file will store your API keys and configuration secrets.

5.  **Configure Environment Variables:** Add the following variables to your `.env` file, replacing the placeholder values with your actual credentials and settings:

    ```dotenv
    # --- Required --- 
    OPENAI_API_KEY="your_openai_api_key"
    
    # --- Ticket System Configuration --- 
    # Set to either "Trello" or "Jira"
    TICKET_SYSTEM="Trello" 
    
    # --- Trello Configuration (Required if TICKET_SYSTEM="Trello") --- 
    TRELLO_API_KEY="your_trello_api_key"
    TRELLO_API_TOKEN="your_trello_api_token" 
    TRELLO_LIST_ID="your_target_trello_list_id" # e.g., 6810b8057c02db1e7c00616c
    # TRELLO_BOARD_ID="your_trello_board_id" # Optional, but good for reference (e.g., DzhpzEl5)
    
    # --- Jira Configuration (Required if TICKET_SYSTEM="Jira") --- 
    # JIRA_URL="https://your-domain.atlassian.net"
    # JIRA_PROJECT_KEY="YOUR_PROJECT_KEY" # e.g., SUP
    # JIRA_ISSUE_TYPE="Task" # Or "Bug", "Support Request", etc.
    # JIRA_USER_EMAIL="your_jira_email@example.com"
    # JIRA_API_TOKEN="your_jira_api_token"
    ```
    *   Get OpenAI API Key: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    *   Get Trello API Key and Token: [https://trello.com/app-key](https://trello.com/app-key) (Generate a Token from this page)
    *   Get Trello List ID: Open your Trello board, add `.json` to the URL (e.g., `https://trello.com/b/BOARD_ID/board-name.json`), and find the `id` for the desired list under the `lists` array.
    *   Get Jira API Token: [https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/)

6.  **Create Data Folder:** Create a directory named `data` in the root of your project.
    ```bash
    mkdir data
    ```

7.  **Add PDF Documents:** Place the PDF files you want to chat with into the `./data` folder.

## Running the Application

Once the setup is complete, you can run the Streamlit application from your terminal:

```bash
streamlit run streamlit_app_fixed.py
```

The application will open in your web browser.

## Usage

1.  **Chat:** Type your questions about the documents in the chat input box at the bottom of the page.
2.  **Upload Documents:** Use the file uploader in the sidebar to add new PDF documents to the `./data` folder. The application will need to re-index the documents (this happens automatically on the next query or page refresh after upload).
3.  **Ticket Creation:** If you ask a question and the system responds that it couldn't find the answer, a form will appear allowing you to enter your name, email, a summary, and a description to create a support ticket in the configured system (Trello or Jira).

## Project Structure

```
.
├── streamlit_app_fixed.py  # Main application script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, config)
├── data/                   # Folder for your PDF documents
│   └── example.pdf
├── vector_store/           # Folder where the FAISS index is stored (auto-generated)
│   ├── index.faiss
│   └── index.pkl
└── README.md               # This file
```

## Dependencies

All required Python packages are listed in `requirements.txt`.

