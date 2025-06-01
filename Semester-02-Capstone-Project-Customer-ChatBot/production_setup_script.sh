#!/bin/bash
# Customer Support RAG Chatbot - Production Setup Script

echo "🚀 Customer Support RAG Chatbot Setup"
echo "======================================"

echo ""
echo "📋 Prerequisites:"
echo "1. GitHub account with repository for support tickets"
echo "2. Python 3.10+ installed"
echo "3. Git installed"

echo ""
echo "🔧 Step 1: Create GitHub Personal Access Token"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Set name: 'Customer Support Chatbot'"
echo "4. Select scope: 'repo' (full control)"
echo "5. Click 'Generate token'"
echo "6. Copy the token (starts with ghp_)"

echo ""
read -p "📝 Enter your GitHub token: " GITHUB_TOKEN

echo ""
echo "🔧 Step 2: Repository Setup"
echo "Choose an option:"
echo "1. Use existing repository"
echo "2. Create new repository"

read -p "Enter choice (1 or 2): " REPO_CHOICE

if [ "$REPO_CHOICE" = "1" ]; then
    read -p "📝 Enter repository (username/repo-name): " GITHUB_REPO
elif [ "$REPO_CHOICE" = "2" ]; then
    read -p "📝 Enter repository name: " REPO_NAME
    
    echo "Creating repository '$REPO_NAME'..."
    curl -H "Authorization: token $GITHUB_TOKEN" \
         -d "{\"name\":\"$REPO_NAME\",\"description\":\"Customer Support Tickets\",\"has_issues\":true}" \
         https://api.github.com/user/repos
    
    # Extract username from token
    USERNAME=$(curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep '"login"' | cut -d'"' -f4)
    GITHUB_REPO="$USERNAME/$REPO_NAME"
    echo "✅ Repository created: $GITHUB_REPO"
fi

echo ""
echo "🔧 Step 3: Company Information"
read -p "📝 Company Name: " COMPANY_NAME
read -p "📝 Support Email: " COMPANY_EMAIL
read -p "📝 Support Phone: " COMPANY_PHONE
read -p "📝 Website URL: " COMPANY_WEBSITE

echo ""
echo "🔧 Step 4: Configuration"

# Create .env file
cat > .env << EOF
# GitHub Integration
GITHUB_TOKEN=$GITHUB_TOKEN
GITHUB_REPO=$GITHUB_REPO

# Company Information
COMPANY_NAME=$COMPANY_NAME
COMPANY_EMAIL=$COMPANY_EMAIL
COMPANY_PHONE=$COMPANY_PHONE
COMPANY_WEBSITE=$COMPANY_WEBSITE
COMPANY_ADDRESS=123 Business St, City, State 12345

# AI Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=data/vector_store
LOG_LEVEL=INFO
USE_GRADIO=False
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.3
EOF

echo "✅ Configuration saved to .env"

echo ""
echo "🔧 Step 5: Install Dependencies"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "🔧 Step 6: Process Documents"
echo "📁 Add your documents to data/raw/ directory"
echo "   - At least 3 documents total"
echo "   - At least 2 PDFs"
echo "   - At least 1 PDF with 400+ pages"

read -p "Press Enter when documents are ready..."

python run_ingest.py

echo ""
echo "🔧 Step 7: Test System"
python -c "
import sys; sys.path.insert(0, 'src')
from services.chatbot import ChatBot
from services.ticket_service import TicketService

bot = ChatBot()
tickets = TicketService()

print(f'✅ ChatBot Ready: {bot.is_ready()}')
print(f'✅ GitHub Connected: {tickets.test_connection()}')

if bot.is_ready():
    stats = bot.get_vector_store_stats()
    print(f'📊 Documents: {stats[\"unique_sources\"]}')
    print(f'📊 Vectors: {stats[\"total_vectors\"]}')
"

echo ""
echo "🚀 Step 8: Launch Application"
echo "Run: streamlit run main.py"
echo "Or:  python main.py"

echo ""
echo "🎉 Setup Complete!"
echo "📱 Access your chatbot at: http://localhost:8502"
echo "🎫 Support tickets will be created at: https://github.com/$GITHUB_REPO/issues"

echo ""
echo "⚠️  SECURITY NOTES:"
echo "- Keep your GitHub token secure"
echo "- Add .env to .gitignore"
echo "- Regenerate token if compromised"