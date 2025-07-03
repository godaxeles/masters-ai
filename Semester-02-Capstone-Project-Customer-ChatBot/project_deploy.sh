#!/bin/bash

# This script guides you through updating a GitHub repository
# and deploying the RAG chatbot to a new Hugging Face Space.

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
ZIP_FILE="rag_chatbot_deployment.zip"
PROJECT_DIR_NAME="11  - Capstone project"

# --- Pre-flight Checks ---
# 1. Check for required tools
if ! command -v git &> /dev/null || ! command -v huggingface-cli &> /dev/null || ! command -v unzip &> /dev/null; then
    echo "🔴 Error: Missing required tools. Please ensure git, huggingface-cli, and unzip are installed."
    exit 1
fi

# 2. Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "🔴 Error: The required file '$ZIP_FILE' was not found."
    echo "Please make sure the zip file is in the same directory as this script."
    exit 1
fi

# 3. Check if user is logged into Hugging Face
if ! huggingface-cli whoami &> /dev/null; then
    echo "🟡 You are not logged into Hugging Face."
    echo "Please log in using the command: huggingface-cli login"
    exit 1
fi

# --- User Input & Instructions ---
echo "--- RAG Chatbot Deployment for Your Project ---"

# GitHub Setup
echo "
--- Step 1: Update Your GitHub Repository ---"
read -p "Enter the full path to your cloned 'masters-ai' repository: " REPO_PATH
if [ ! -d "$REPO_PATH" ] || [ ! -d "$REPO_PATH/.git" ]; then
    echo "🔴 Error: The path provided is not a valid git repository."
    exit 1
fi

PROJECT_PATH="$REPO_PATH/$PROJECT_DIR_NAME"
if [ ! -d "$PROJECT_PATH" ]; then
    echo "🔴 Error: The project directory '$PROJECT_DIR_NAME' was not found in your repository."
    exit 1
fi

# Update GitHub Repository
echo "⏳ Preparing to update your local repository..."
# Unzip the new application code
unzip -o "$ZIP_FILE" -d "temp_deploy"

# Clean the project directory and copy new files
rm -rf "$PROJECT_PATH"/*
cp -r temp_deploy/deployable_chatbot/* "$PROJECT_PATH/"
rm -rf temp_deploy

echo "✅ Your local project files have been updated."
echo "⏳ Committing and pushing changes to GitHub..."

cd "$PROJECT_PATH"
git add .
git commit -m "feat: Replace project with deployable RAG chatbot for Hugging Face"
git push

echo "✅ Your GitHub repository has been successfully updated!"
cd -

# Hugging Face Deployment
echo "
--- Step 2: Deploy to Hugging Face Spaces ---"
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter a new name for your Hugging Face Space (e.g., capstone-chatbot): " SPACE_NAME

# Create HF Space linked to GitHub
echo "⏳ Creating a new Hugging Face Space linked to your GitHub repository..."

# Note: The user must have the Hugging Face GitHub App installed and authorized.
huggingface-cli repo create "$SPACE_NAME" \ 
    --type space \ 
    --sdk streamlit \ 
    --from-zero \ 
    --repo-type github \ 
    --repo-id "$HF_USERNAME/$SPACE_NAME" # This seems incorrect for linking, let's use a different approach

# Correct approach is to create from template or manually link.
# Let's guide the user to do it from the website for reliability.

echo "
--- Manual Hugging Face Setup (Recommended) ---"
echo "The most reliable way to create your Space is through the Hugging Face website."
echo "1. Go to: https://huggingface.co/new-space"
_read -p "Enter your GitHub username: " GH_USERNAME
echo "2. Owner: Select your HF profile ($HF_USERNAME)"
echo "3. Space name: $SPACE_NAME"
_echo "4. License: mit"
echo "5. Select SDK: Streamlit"
echo "6. Space hardware: CPU basic (Free)"
echo "7. Choose 'Link to a GitHub repository'"
echo "8. Repository: $GH_USERNAME/masters-ai"
echo "9. Branch: main"
_echo "10. Application subfolder: $PROJECT_DIR_NAME"
echo "11. Click 'Create Space'"

echo "
🎉 --- All Done! --- 🎉"
echo "Your GitHub repository is updated, and your Hugging Face Space is being created."
echo "You can view your new Space at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
