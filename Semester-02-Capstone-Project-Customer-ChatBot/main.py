#!/usr/bin/env python3
"""
Main entry point for the Customer Support RAG Chatbot.
This file can be run directly with Streamlit: streamlit run main.py
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main application
from app import main

if __name__ == "__main__":
    main()
