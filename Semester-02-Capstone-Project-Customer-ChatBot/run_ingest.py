#!/usr/bin/env python3
"""
Standalone document ingestion script with proper import handling.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now run the ingestion
from ingest import main

if __name__ == "__main__":
    main()
