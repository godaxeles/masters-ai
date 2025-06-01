"""Configuration management for the Customer Support RAG Chatbot."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # GitHub Integration
    github_token: str = Field(..., description="GitHub personal access token")
    github_repo: str = Field(..., description="GitHub repository (user/repo)")

    # OpenAI (Optional)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")

    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model to use"
    )
    vector_store_path: Path = Field(
        default=Path("data/vector_store"),
        description="Path to vector store"
    )

    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    use_gradio: bool = Field(default=False, description="Use Gradio instead of Streamlit")
    max_chunk_size: int = Field(default=512, description="Maximum chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Chunk overlap in tokens")
    top_k_retrieval: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity threshold for retrieval"
    )

    # Company Information
    company_name: str = Field(default="Your Company", description="Company name")
    company_email: str = Field(
        default="support@yourcompany.com",
        description="Company support email"
    )
    company_phone: str = Field(
        default="+1-555-0123",
        description="Company support phone"
    )
    company_website: str = Field(
        default="https://yourcompany.com",
        description="Company website"
    )
    company_address: str = Field(
        default="123 Business St, City, State 12345",
        description="Company address"
    )

    # Data Paths
    data_raw_path: Path = Field(
        default=Path("data/raw"),
        description="Path to raw documents"
    )
    data_processed_path: Path = Field(
        default=Path("data/processed"),
        description="Path to processed documents"
    )
    logs_path: Path = Field(
        default=Path("logs"),
        description="Path to log files"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
settings.data_raw_path.mkdir(parents=True, exist_ok=True)
settings.data_processed_path.mkdir(parents=True, exist_ok=True)
settings.vector_store_path.mkdir(parents=True, exist_ok=True)
settings.logs_path.mkdir(parents=True, exist_ok=True)